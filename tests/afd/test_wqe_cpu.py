"""CPU wire/scheduling model, not execution of the Ascend C implementation."""
import struct
import unittest


SQ_DEPTH = 32768
TASK_WRITES = 8
MAX_PENDING = 32
WRITE_HEADER = 0x0000030030010000
FINAL_FLAGS = (1 << 18) | (1 << 19) | (1 << 21)
U32_MASK = (1 << 32) - 1


def encode_task(head, requests, num_sges):
    """Model native warp/word placement including a split two-BB WQE."""
    ring = {}
    blocks = 1 if num_sges == 1 else 2
    for token, (peer, offset, src, meta, hidden_bytes, meta_bytes) in enumerate(requests):
        bb = (head + token * blocks) & U32_MASK
        words = [0] * (blocks * 8)
        words[0] = WRITE_HEADER | FINAL_FLAGS | (int(not (bb & SQ_DEPTH)) << 31) | (bb & (SQ_DEPTH - 1))
        words[1:6] = peer[:5]
        words[1] |= num_sges << 24
        words[5] += offset
        words[6:8] = hidden_bytes, src
        if num_sges == 2:
            words[8:10] = meta_bytes, meta
        for lane, word in enumerate(words):
            index = ((bb + lane // 8) & (SQ_DEPTH - 1)) * 8 + lane % 8
            if index in ring:
                raise AssertionError('overwritten SQ word')
            ring[index] = word
    return ring


def submission_groups(count):
    """Tasks fill at eight; only committed groups may be drained."""
    pending, groups = 0, []
    for start in range(0, count, TASK_WRITES):
        size = min(TASK_WRITES, count - start)
        if pending + size > MAX_PENDING:
            groups.append(pending)
            pending = 0
        pending += size
        if pending == MAX_PENDING:
            groups.append(pending)
            pending = 0
    if pending:
        groups.append(pending)
    return groups


class WqeTests(unittest.TestCase):
    def test_native_header_sges_and_wrap(self):
        # Include an odd SQ position left by scalar control writes and uint32 wrap.
        for num_sges in (1, 2):
            blocks = num_sges
            for head in (0, 1, SQ_DEPTH-1, SQ_DEPTH, 2*SQ_DEPTH-1, U32_MASK-2):
                for count in (1, 3, 8):
                    requests = [([0x123 | (0x456 << 32), 0x111, 0x222, 0x333, 1 << 40],
                                 i*8192, (1 << 42)+i*7168, (1 << 44)+i*256, 7168, 256)
                                for i in range(count)]
                    ring = encode_task(head, requests, num_sges)
                    for i, (peer, offset, src, meta, hidden_bytes, meta_bytes) in enumerate(requests):
                        bb = (head+i*blocks) & U32_MASK
                        data = b''.join(struct.pack('<Q', ring[((bb+j//8) & (SQ_DEPTH-1))*8+j%8])
                                        for j in range(blocks*8))
                        # Independent scalar field decoder of native SQE ABI.
                        header, transport = struct.unpack_from('<QQ', data)
                        self.assertEqual(header & 0xffff, bb & (SQ_DEPTH-1))
                        self.assertEqual((header >> 16) & 7, 5)
                        self.assertEqual((header >> 19) & 1, 1)
                        self.assertEqual((header >> 21) & 1, 1)
                        self.assertEqual((header >> 31) & 1, int(not(bb & SQ_DEPTH)))
                        self.assertEqual((header >> 40) & 0xff, 3)
                        self.assertEqual((transport >> 24) & 0xff, num_sges)
                        self.assertEqual(struct.unpack_from('<Q', data, 40)[0], peer[4]+offset)
                        self.assertEqual(struct.unpack_from('<IIQ', data, 48), (hidden_bytes, 0, src))
                        if num_sges == 2:
                            self.assertEqual(struct.unpack_from('<IIQ', data, 64), (meta_bytes, 0, meta))
                            self.assertEqual(data[80:], bytes(48))

    def test_zero_tail_and_cq_budget(self):
        for count in (0, 1, 7, 8, 9, 31, 32, 33, 257, 32768):
            groups = submission_groups(count)
            self.assertEqual(sum(groups), count)
            self.assertTrue(all(0 < n <= MAX_PENDING for n in groups))
            # At minimum accepted power-of-two CQ depth (64), no overwritten CQE.
            self.assertTrue(all(n < 64 and 2*n < SQ_DEPTH for n in groups))
            self.assertEqual(len(groups), (count+MAX_PENDING-1)//MAX_PENDING)


if __name__ == '__main__':
    unittest.main()
