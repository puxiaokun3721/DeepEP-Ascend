"""CPU oracle and source-contract tests. These do NOT execute Ascend kernels."""
import ctypes
from pathlib import Path
import re
import struct
import unittest
import numpy as np
from golden import golden, layout, verify_snapshot, InputLane

ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / 'deep_ep/include/deep_ep/layout/afd/context.hpp'


def parse_structs(text):
    """Read real implementation declarations, reject unsupported C++ grammar."""
    text = re.sub(r'//[^\n]*|/\*.*?\*/', '', text, flags=re.S)
    types = {'uint32_t': ctypes.c_uint32, 'uint64_t': ctypes.c_uint64}
    result = {}
    for name, body in re.findall(r'struct alignas\(32\) (\w+) \{(.*?)\};', text, re.S):
        fields = []
        for declaration in body.split(';'):
            if not declaration.strip():
                continue
            kind, names = declaration.strip().split(None, 1)
            for field in names.split(','):
                match = re.fullmatch(r'(\w+)(?:\[(\d+)\])?', field.strip())
                if not match:
                    raise ValueError(declaration)
                field_type = types[kind]
                if match[2]:
                    field_type *= int(match[2])
                fields.append((match[1], field_type))
        # ctypes does not inherit C++ alignas(32) for nested wire structures.
        # Materialize that padding so both member offsets and tail size match.
        aligned_fields = []
        cursor = 0
        for field_name, field_type in fields:
            alignment = 32 if field_type in result.values() else ctypes.alignment(field_type)
            padding = (-cursor) % alignment
            if padding:
                aligned_fields.append((f'_padding_{len(aligned_fields)}', ctypes.c_uint8 * padding))
                cursor += padding
            aligned_fields.append((field_name, field_type))
            cursor += ctypes.sizeof(field_type)
        if cursor % 32:
            aligned_fields.append(('_tail_padding', ctypes.c_uint8 * ((-cursor) % 32)))
        typ = type(name, (ctypes.LittleEndianStructure,), {'_fields_': aligned_fields})
        types[name] = typ
        result[name] = typ
    return result


def source_workspace(c, t, k, e, n, m, slots):
    """Evaluate only actual make_workspace arithmetic (no substitute formula)."""
    text = HEADER.read_text(encoding='utf-8-sig')
    body = text.split('AFDWorkspace w{};', 1)[1].split('return w;', 1)[0]
    env = dict(c=c, t=t, k=k, e=e, n=n, m=m, slots=slots, p=0, j=(c+t-1)//t,
               align32=lambda n: (n+31)//32*32, uint64_t=int)
    out = {}
    for statement in body.split(';')[1:]:
        statement = statement.strip()
        if statement.startswith('w.'):
            name, expression = statement[2:].split('=')
            out[name] = eval(expression, {'__builtins__': {}}, env)
        elif statement.startswith('p+='):
            env['p'] += eval(statement[3:], {'__builtins__': {}}, env)
        elif statement:
            raise ValueError(statement)
    return out


class ABIContract(unittest.TestCase):
    def test_actual_header_struct_sizes_and_offsets(self):
        types = parse_structs(HEADER.read_text(encoding='utf-8-sig'))
        expected = {'AFDControlBlock':32, 'AFDTaskInfo':64, 'AFDInbox':64,
                    'AFDRoundLedger':64, 'AFDSendState':32, 'AFDLane':32,
                    'AFDPeer':32, 'AFDAddress':64, 'AFDWorkspace':96,
                    'AFDContextLayout':256}
        for name, size in expected.items():
            with self.subTest(name=name):
                self.assertEqual(ctypes.sizeof(types[name]), size)
        self.assertEqual(types['AFDContextLayout'].workspace.offset, 160)
        self.assertEqual(types['AFDContextLayout'].channel_stride.offset, 64)
        self.assertEqual(types['AFDContextLayout'].lanes_offset.offset, 72)
        self.assertEqual(types['AFDPeer'].control_base.offset, 16)
        self.assertEqual(types['AFDInbox'].num_recv_records.offset, 24)
        self.assertEqual(types['AFDTaskInfo'].num_comm_rounds.offset, 24)
        self.assertEqual(types['AFDSendState'].last_input_bit.offset, 12)
        task = types['AFDTaskInfo']()
        task.abi_version, task.num_comm_rounds = 2, 7
        self.assertEqual(bytes(task), struct.pack('<16I', 2,0,0,0,0,0,7,*([0]*9)))

    def test_implementation_workspace_against_independent_budget(self):
        for c in (1,32,256,32768):
            for t in (32,64,128,256):
                for m in (128,352):
                    with self.subTest(c=c,t=t,m=m):
                        w = source_workspace(c,t,8,64,8,m,3)
                        lengths = [4*c*8,c*m,4*((c+t-1)//t)*72,4*((c+t-1)//t)*72,
                                   4*c*8,256,256,32,96,256]
                        names = ['dst_buffer_slot_idx','metadata_send_buffer','tile_counts',
                                 'tile_psum','route_reference_send','expert_counts',
                                 'expert_prefix','rank_totals','send_state','error_scratch']
                        cursor = 0
                        for name, length in zip(names,lengths):
                            self.assertEqual(w[name], cursor)
                            self.assertEqual(w[name] % 32, 0)
                            cursor += (length+31)//32*32
                        self.assertEqual(w['num_bytes'], cursor)

    def test_record_layout_source_expressions(self):
        text = (HEADER.parent/'token.hpp').read_text(encoding='utf-8-sig')
        methods = dict(re.findall(r'uint64_t (\w+)\(\) const \{ return (.*?); \}',text))
        for hidden, sf, k in ((14336,0,8),(7168,224,8),(256,0,1),(128,4,32)):
            env = dict(num_hidden_bytes=hidden,num_sf_bytes=sf,num_topk=k,
                       align32=lambda n:(n+31)//32*32,align512=lambda n:(n+511)//512*512)
            for name, expression in methods.items():
                env[name] = lambda expression=expression: eval(expression, {'__builtins__':{}},env)
            expected = layout(hidden,sf,k)
            for key, method in [('sf','sf_offset'),('idx','topk_idx_offset'),('weights','topk_weights_offset'),
                                ('token','src_token_idx_offset'),('valid','valid_topk_mask_offset'),
                                ('metadata','metadata_stride'),('stride','record_stride')]:
                self.assertEqual(env[method](),expected[key])


class OracleTests(unittest.TestCase):
    def test_dedup_and_mask(self):
        records, routes = golden(np.array([[0,1,2],[3,4,5],[-1,99,-1]]),6,2,[1,1,0])
        self.assertEqual(records,[{0},{1}])
        self.assertEqual(sum(sum(r.values()) for r in routes),6)

    def test_rejected_routes_and_dimensions(self):
        for row,e,n in (([-1],8,2),([8],8,2),([1,1],8,2),([1],8,0),([0],0,1),([0],7,2),([0,1],1,1)):
            with self.subTest(row=row,e=e,n=n), self.assertRaises(ValueError):
                golden(np.array([row]),e,n)

    def test_boundaries_and_zero_routes(self):
        for b in (0,1,31,32,33,255,256,257,32767,32768,32769,98305):
            rows = np.tile(np.arange(8),(b,1))
            records,routes = golden(rows,64,8)
            self.assertEqual(len(records[0]),b)
            self.assertTrue(all(not r for r in records[1:]))
            self.assertEqual(sum(sum(r.values()) for r in routes),b*8)

    def test_snapshot_oracle_accepts_permutation_and_detects_corruption(self):
        topk = np.array([[0,1],[1,2]],dtype='<i8')
        weights = np.array([[0x7fc01234,0x80000000],[0x7f800000,1]],dtype='<u4')
        x = np.arange(512,dtype=np.uint16).view(np.uint8).reshape(2,512)
        sf = np.zeros((2,0),dtype=np.uint8)
        pos = layout(512,0,2)
        payload = bytearray(2*pos['stride'])
        for rec,tok in enumerate((1,0)):
            start = rec*pos['stride']
            payload[start:start+512] = x[tok].tobytes()
            payload[start+pos['idx']:start+pos['idx']+16] = topk[tok].tobytes()
            payload[start+pos['weights']:start+pos['weights']+8] = weights[tok].tobytes()
            struct.pack_into('<II',payload,start+pos['token'],tok,3)
        args = ([2,3,0,1],[1,2,1],[1,3,4],0,x,sf,topk,weights,4,1)
        # Four local experts: final expert has no routes.
        args = (args[0],args[1]+[0],args[2]+[4])+args[3:]
        self.assertEqual(verify_snapshot(bytes(payload),*args)['route_errors'],0)
        payload[0] ^= 1
        with self.assertRaisesRegex(AssertionError,'hidden bits'):
            verify_snapshot(bytes(payload),*args)

    def test_fp8_sf_special_bits_and_all_snapshot_outputs(self):
        topk = np.array([[0]],dtype='<i8')
        weights = np.array([[0x7fc01234]],dtype='<u4')
        x = np.arange(128,dtype=np.uint8).reshape(1,128)
        sf_storage = np.array([[0x8000],[0x7fff]],dtype='<u2')
        sf = sf_storage.T.copy().view(np.uint8).reshape(1,4)
        pos = layout(128,4,1)
        payload = bytearray(pos['stride'])
        payload[:128] = x[0].tobytes()
        payload[pos['sf']:pos['sf']+4] = sf[0].tobytes()
        payload[pos['idx']:pos['idx']+8] = topk[0].tobytes()
        payload[pos['weights']:pos['weights']+4] = weights[0].tobytes()
        struct.pack_into('<II',payload,pos['token'],0,1)
        args = ([0],[1],[1],0,x,sf,topk,weights,1,1)
        verify_snapshot(bytes(payload),*args)
        for offset, label in ((pos['sf'],'SF bits'),(pos['idx'],'topk bits'),
                              (pos['weights'],'weight bits'),(pos['valid'],'valid mask'),
                              (pos['sf']+4,'metadata padding')):
            with self.subTest(label=label):
                bad = bytearray(payload); bad[offset] ^= 1
                with self.assertRaisesRegex(AssertionError,label):
                    verify_snapshot(bytes(bad),*args)
        for refs,counts,prefix in (([1],[1],[1]),([-1],[1],[1]),([2**32],[1],[1]),([0],[0],[0]),([0],[1],[0])):
            with self.subTest(refs=refs,counts=counts,prefix=prefix), self.assertRaises(AssertionError):
                verify_snapshot(bytes(payload),refs,counts,prefix,*args[3:])


class ProtocolSpecification(unittest.TestCase):
    def test_three_rounds_delayed_ack_and_ledger(self):
        lane = InputLane(3)
        self.assertEqual(lane.publish(0),0)  # Includes zero-route F.
        lane.ack(0); lane.ack(1)
        with self.assertRaisesRegex(BlockingIOError,'ACK'): lane.publish(1)
        lane.ack(2)
        self.assertEqual(lane.publish(1),1)
        for rank in range(3): lane.ack(rank)
        with self.assertRaisesRegex(BlockingIOError,'Ledger'): lane.publish(2)
        lane.recv_release(0)
        self.assertEqual(lane.publish(2),0)

    def test_cross_transaction_and_independent_decode_slots(self):
        lanes = [InputLane(2) for _ in range(3)]
        for lane in lanes: self.assertEqual(lane.publish(0),0)
        for lane in lanes:
            lane.recv_release(0)
            with self.assertRaises(BlockingIOError): lane.publish(0)
            lane.ack(0); lane.ack(1)
            self.assertEqual(lane.publish(0),1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
