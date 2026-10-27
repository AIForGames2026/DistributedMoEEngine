# Wire Format v0

Status: **frozen for the PoC** (TASK-06). Any change bumps `TF_WIRE_VERSION` / `wire.version`.

The format is defined twice, in [`src/wire.zig`](../src/wire.zig) and [`include/tf_wire.h`](../include/tf_wire.h). Both assert every size and offset below at compile time, and `zig build test` checks that the two agree at run time (`tf_wire_layout`).

## Token packet (7,552 bytes)

One packet carries one token's hidden state to one expert (dispatch) or one expert's output back to its origin (combine). It is written whole into a fixed-size slot of the receiver's ring. All integers are little-endian; every machine in the project (GB10, x86_64) is little-endian, so the structs are sent as-is.

| Offset | Size | Field | Type | Notes |
|---|---|---|---|---|
| `0x0000` | 8 | `token_id` | `u64` | Global sequence index |
| `0x0008` | 8 | `generation` | `u64` | Full slot generation, starts at 1, +1 each time the slot is reused |
| `0x0010` | 2 | `src_rank` | `u16` | Sending node |
| `0x0012` | 2 | `layer` | `u16` | Target layer |
| `0x0014` | 2 | `expert` | `u16` | Target expert (global index) |
| `0x0016` | 2 | `flags` | `u16` | Reserved, 0 in v0 |
| `0x0018` | 4 | `gate_weight` | `f32` | Top-k gate weight, applied on combine |
| `0x001C` | 4 | `payload_len` | `u32` | Valid bytes in `payload` |
| `0x0020` | 7,168 | `payload` | `fp8_e4m3[7168]` | Hidden state; unused tail bytes are zero |
| `0x1C20` | 224 | `scales` | `e8m0[224]` | One scale per 32-element block |
| `0x1D00` | 112 | `pad` | `u8[112]` | Zero; brings the packet to 59 x 128 bytes |
| `0x1D70` | 4 | `tail.crc32c` | `u32` | CRC32C of bytes `0x0000` to `0x1D6F` |
| `0x1D74` | 4 | `tail.generation_lo` | `u32` | Low 32 bits of `generation` |
| `0x1D78` | 8 | `tail.reserved` | `u8[8]` | Zero |

### Decisions

- **7,552 bytes, not the blueprint's 7,424.** The research blueprint lists `fp8_e8m0[224]` scales but gives them only 208 bytes (`0x1C20` to `0x1CEF`). With 224 scales the packet is 32 + 7,168 + 224 + 16 = 7,440 bytes. That is padded to 7,552 (59 x 128) so every slot starts on a 128-byte boundary. This is the size measured on `gx10-03`: 3.26 us per packet in RDMA loopback ([probe](../bench/platform/gx10-03-2026-10-26.md)).
- **7,168-byte payload.** This fits DeepSeek-V3-class hidden sizes (7,168) at one byte per element. Smaller models, like OLMoE-1B-7B (2,048), set `payload_len` and leave the rest zero. The engine can shrink the slot size per model later; that becomes v1.
- **Full 64-bit generation.** An 8-bit epoch wraps after 256 reuses of a slot, which happens within seconds. The full value lives in the header. The tail repeats its low 32 bits and is written last, so a receiver that polls memory (instead of waiting for a completion) can tell a fully placed packet from a stale one.
- **CRC32C** (Castagnoli, the iSCSI polynomial) covers everything before the tail. Check value: `CRC32C("123456789") = 0xE3069283`.

## Immediate data (32 bits)

Sent with every RDMA `WRITE_WITH_IMM`, and carried in the frame header by the TCP and shared-memory backends, so all transports look the same to the engine.

| Bits | Field | Meaning |
|---|---|---|
| 0 to 15 | `slot` | Slot index in the receiver's ring (up to 65,536 slots) |
| 16 to 17 | `kind` | 0 dispatch, 1 combine, 2 credit grant, 3 control |
| 18 to 25 | `gen8` | Low 8 bits of the slot generation |
| 26 to 31 | `src_rank` | Sending node (up to 64) |

The two research documents proposed different immediate layouts. Both packed token, expert, and routing data into the immediate, which duplicates the header and caps experts at 256. v0 instead makes the immediate answer only "which slot just landed, from whom, and is it the generation I expect?" Everything else is read from the header. `gen8` is a quick check before the receiver touches the slot; the authoritative check is the full `generation` plus the CRC.

C helpers: `TF_IMM(slot, kind, gen8, src)` and `TF_IMM_SLOT/KIND/GEN8/SRC(imm)`. Zig: `wire.Imm` (a `packed struct(u32)`).

## Receiver checks

On every packet, in order:

1. `imm.kind` is valid and `imm.slot` is inside the ring.
2. `imm.gen8` matches the expected generation for that slot.
3. `header.generation` and `tail.generation_lo` both match the expected generation (`Packet.check`).
4. `tail.crc32c` matches (`Packet.check`).

A failure in 2 or 3 is a protocol bug (a sender overwrote a slot it had no credit for). A failure in 4 is corruption. Both are counted and logged; v0 drops the packet and fails the test run.
