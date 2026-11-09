// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "sw/utils/f2_loader/ocl_socket.h"

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

namespace f2 {

std::unique_ptr<SocketOcl> SocketOcl::Connect(const std::string &path, double timeout_s,
                                              std::string *error) {
  sockaddr_un sa = {};
  sa.sun_family  = AF_UNIX;
  if (path.size() >= sizeof(sa.sun_path)) {
    *error = "socket path too long: " + path;
    return nullptr;
  }
  strncpy(sa.sun_path, path.c_str(), sizeof(sa.sun_path) - 1);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(timeout_s);
  for (;;) {
    const int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
      *error = std::string("socket: ") + strerror(errno);
      return nullptr;
    }
    if (connect(fd, reinterpret_cast<sockaddr *>(&sa), sizeof(sa)) == 0) {
      return std::unique_ptr<SocketOcl>(new SocketOcl(fd));
    }
    const int err = errno;
    close(fd);
    if (std::chrono::steady_clock::now() > deadline) {
      *error = "cannot connect to " + path + ": " + strerror(err);
      return nullptr;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

SocketOcl::~SocketOcl() {
  Request(kSocketOpDone, 0, 0);
  close(fd_);
}

uint32_t SocketOcl::Request(uint32_t op, uint32_t offset, uint32_t data) {
  uint8_t buf[12];
  const uint32_t words[3] = {op, offset, data};
  for (int w = 0; w < 3; ++w)
    for (int i = 0; i < 4; ++i)
      buf[w * 4 + i] = static_cast<uint8_t>(words[w] >> (8 * i));
  for (size_t sent = 0; sent < sizeof(buf);) {
    const ssize_t n = send(fd_, buf + sent, sizeof(buf) - sent, MSG_NOSIGNAL);
    if (n < 0 && errno == EINTR)
      continue;
    if (n <= 0) {
      fprintf(stderr, "f2_loader: testbench socket closed (send: %s)\n", strerror(errno));
      exit(2);
    }
    sent += static_cast<size_t>(n);
  }
  uint8_t r[4];
  for (size_t got = 0; got < sizeof(r);) {
    const ssize_t n = recv(fd_, r + got, sizeof(r) - got, 0);
    if (n < 0 && errno == EINTR)
      continue;
    if (n <= 0) {
      if (op == kSocketOpDone)
        return 0;  // the testbench may finish first
      fprintf(stderr, "f2_loader: testbench socket closed (recv)\n");
      exit(2);
    }
    got += static_cast<size_t>(n);
  }
  return static_cast<uint32_t>(r[0]) | static_cast<uint32_t>(r[1]) << 8 |
         static_cast<uint32_t>(r[2]) << 16 | static_cast<uint32_t>(r[3]) << 24;
}

uint32_t SocketOcl::Read32(uint32_t offset) { return Request(kSocketOpRead, offset, 0); }

void SocketOcl::Write32(uint32_t offset, uint32_t value) { Request(kSocketOpWrite, offset, value); }

void SocketOcl::SleepUs(uint32_t us) {
  constexpr uint32_t kMaxUs = 0xFFFFFFFFu / kClkMainMhz;
  Request(kSocketOpWait, 0, (us > kMaxUs ? kMaxUs : us) * kClkMainMhz);
}

}  // namespace f2
