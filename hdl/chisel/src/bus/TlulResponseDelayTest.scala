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
import chisel3.simulator.scalatest.ChiselSim
import org.scalatest.freespec.AnyFreeSpec

class TlulResponseDelayTest extends AnyFreeSpec with ChiselSim {
  val readLatency  = 5
  val writeLatency = 3

  def newDut() = new TlulResponseDelay(UInt(32.W), readLatency, writeLatency, depth = 8)

  def reset(dut: TlulResponseDelay[UInt]): Unit = {
    dut.io.enq.valid.poke(false.B)
    dut.io.deq.ready.poke(true.B)
    dut.reset.poke(true.B)
    dut.clock.step()
    dut.reset.poke(false.B)
  }

  // Accepts one response and returns how many cycles pass until it shows up on the output.
  def delayOf(dut: TlulResponseDelay[UInt], isRead: Boolean, value: Int): Int = {
    dut.io.enq.valid.poke(true.B)
    dut.io.enq.bits.poke(value.U)
    dut.io.isRead.poke(isRead.B)
    dut.clock.step()
    dut.io.enq.valid.poke(false.B)
    var cycles = 1
    while (dut.io.deq.valid.peek().litValue == 0) {
      dut.clock.step()
      cycles += 1
      assert(cycles < 100, "response never arrived")
    }
    assert(dut.io.deq.bits.peek().litValue == value)
    dut.clock.step() // consume it
    cycles
  }

  "TlulResponseDelay delays a read by the read latency" in {
    simulate(newDut()) { dut =>
      reset(dut)
      assert(delayOf(dut, isRead = true, 0xa5) == readLatency)
    }
  }

  "TlulResponseDelay delays a write by the write latency" in {
    simulate(newDut()) { dut =>
      reset(dut)
      assert(delayOf(dut, isRead = false, 0x5a) == writeLatency)
    }
  }

  "TlulResponseDelay keeps one response per cycle in flight" in {
    simulate(newDut()) { dut =>
      reset(dut)
      val n = 6
      // Send n reads back to back (one per cycle) and watch the output every cycle. Cycle 0 is the
      // first cycle a request is on the input; it is accepted at the end of that cycle.
      var arrivals = Seq[(Int, BigInt)]()
      for (cycle <- 0 until n + readLatency + 4) {
        if (cycle < n) {
          dut.io.enq.valid.poke(true.B)
          dut.io.enq.bits.poke((100 + cycle).U)
          dut.io.isRead.poke(true.B)
          assert(dut.io.enq.ready.peek().litValue == 1, s"request $cycle was not accepted")
        } else {
          dut.io.enq.valid.poke(false.B)
        }
        if (dut.io.deq.valid.peek().litValue == 1) {
          arrivals :+= (cycle, dut.io.deq.bits.peek().litValue)
        }
        dut.clock.step()
      }
      assert(arrivals.map(_._2) == (0 until n).map(i => BigInt(100 + i)), s"got $arrivals")
      // Request i is accepted at the end of cycle i and is visible readLatency cycles later, at
      // the start of cycle i + readLatency, so the stream is one response per cycle.
      assert(arrivals.map(_._1) == (0 until n).map(_ + readLatency), s"got $arrivals")
    }
  }

  "TlulResponseDelay keeps responses in order across reads and writes" in {
    simulate(newDut()) { dut =>
      reset(dut)
      // A slow read followed by a fast write: the write must not overtake it.
      dut.io.enq.valid.poke(true.B)
      dut.io.enq.bits.poke(1.U)
      dut.io.isRead.poke(true.B)
      dut.clock.step()
      dut.io.enq.bits.poke(2.U)
      dut.io.isRead.poke(false.B)
      dut.clock.step()
      dut.io.enq.valid.poke(false.B)
      var seen = Seq[BigInt]()
      var i    = 0
      while (seen.size < 2) {
        if (dut.io.deq.valid.peek().litValue == 1) seen :+= dut.io.deq.bits.peek().litValue
        dut.clock.step()
        i += 1
        assert(i < 100, "responses never arrived")
      }
      assert(seen == Seq(BigInt(1), BigInt(2)))
    }
  }

  "TlulResponseDelay holds a response while the consumer is not ready" in {
    simulate(newDut()) { dut =>
      reset(dut)
      dut.io.deq.ready.poke(false.B)
      dut.io.enq.valid.poke(true.B)
      dut.io.enq.bits.poke(7.U)
      dut.io.isRead.poke(true.B)
      dut.clock.step()
      dut.io.enq.valid.poke(false.B)
      dut.clock.step(20)
      assert(dut.io.deq.valid.peek().litValue == 1)
      assert(dut.io.deq.bits.peek().litValue == 7)
      dut.io.deq.ready.poke(true.B)
      dut.clock.step()
      assert(dut.io.deq.valid.peek().litValue == 0)
    }
  }
}
