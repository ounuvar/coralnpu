// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package bus

import chisel3._
import chisel3.util._

class TlulTimedEntry[T <: Data](gen: T, timeBits: Int) extends Bundle {
  val data    = gen.cloneType
  val readyAt = UInt(timeBits.W)
}

/** Holds each response back until a fixed number of cycles have passed since it was accepted.
  *
  * Models the access latency of a slow memory (for example HBM behind the AWS F2 shell) in
  * simulation. Responses stay in order. Up to `depth` responses can be waiting at once, so the
  * stream keeps full throughput as long as `depth` covers the latency.
  *
  * `isRead` selects the latency for the response being accepted: `readLatency` if true,
  * `writeLatency` if false. A response leaves `latency` cycles after it was accepted (at least 1,
  * because of the queue). Not meant for synthesis: it exists to make simulated memory realistic.
  */
class TlulResponseDelay[T <: Data](
  gen: T,
  readLatency: Int,
  writeLatency: Int,
  depth: Int
) extends Module {
  require(readLatency >= 0 && writeLatency >= 0, "latencies must not be negative")
  require(depth >= 1, "depth must be at least 1")

  val io = IO(new Bundle {
    val enq    = Flipped(Decoupled(gen))
    val isRead = Input(Bool())
    val deq    = Decoupled(gen)
  })

  private val timeBits = 32
  val now              = RegInit(0.U(timeBits.W))
  now := now + 1.U

  val q = Module(new Queue(new TlulTimedEntry(gen, timeBits), depth))

  q.io.enq.valid        := io.enq.valid
  io.enq.ready          := q.io.enq.ready
  q.io.enq.bits.data    := io.enq.bits
  q.io.enq.bits.readyAt := now + Mux(io.isRead, readLatency.U, writeLatency.U)

  // `now` has reached `readyAt` when the wrapped difference is non-negative.
  val due = !(now - q.io.deq.bits.readyAt)(timeBits - 1)

  io.deq.valid   := q.io.deq.valid && due
  io.deq.bits    := q.io.deq.bits.data
  q.io.deq.ready := io.deq.ready && due
}
