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

// Testbench side of the host socket (#15): cl_coralnpu_core_tb.sv serves OCL
// requests from f2_loader --sim_socket (sw/utils/f2_loader/ocl_socket.cc)
// over a Unix socket. The calls block, so simulated time stands still while
// the host is between requests.
//
// Each request is three little-endian 32-bit words (op, offset, data); each
// gets one 32-bit reply. The ops are in sw/utils/f2_loader/ocl_socket.h.

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {

int g_listen_fd = -1;
int g_fd        = -1;

bool RecvAll(int fd, uint8_t *buf, size_t len) {
  while (len > 0) {
    const ssize_t n = recv(fd, buf, len, 0);
    if (n < 0 && errno == EINTR)
      continue;
    if (n <= 0)
      return false;
    buf += n;
    len -= static_cast<size_t>(n);
  }
  return true;
}

bool SendAll(int fd, const uint8_t *buf, size_t len) {
  while (len > 0) {
    const ssize_t n = send(fd, buf, len, MSG_NOSIGNAL);
    if (n < 0 && errno == EINTR)
      continue;
    if (n <= 0)
      return false;
    buf += n;
    len -= static_cast<size_t>(n);
  }
  return true;
}

uint32_t Le32(const uint8_t *p) {
  return static_cast<uint32_t>(p[0]) | static_cast<uint32_t>(p[1]) << 8 |
         static_cast<uint32_t>(p[2]) << 16 | static_cast<uint32_t>(p[3]) << 24;
}

}  // namespace

extern "C" {

// Creates the socket at `path` and waits for the host to connect. Returns 0,
// or -1 on error.
int f2_host_socket_open(const char *path) {
  sockaddr_un sa = {};
  sa.sun_family  = AF_UNIX;
  if (strlen(path) >= sizeof(sa.sun_path)) {
    fprintf(stderr, "[f2_host_socket] path too long: %s\n", path);
    return -1;
  }
  strncpy(sa.sun_path, path, sizeof(sa.sun_path) - 1);
  unlink(path);
  g_listen_fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (g_listen_fd < 0 || bind(g_listen_fd, reinterpret_cast<sockaddr *>(&sa), sizeof(sa)) != 0 ||
      listen(g_listen_fd, 1) != 0) {
    fprintf(stderr, "[f2_host_socket] cannot listen on %s: %s\n", path, strerror(errno));
    return -1;
  }
  fprintf(stderr, "[f2_host_socket] waiting for the host on %s\n", path);
  g_fd = accept(g_listen_fd, nullptr, nullptr);
  if (g_fd < 0) {
    fprintf(stderr, "[f2_host_socket] accept: %s\n", strerror(errno));
    return -1;
  }
  return 0;
}

// Blocks for the next request. Returns 0, or -1 when the host disconnected.
int f2_host_socket_next(uint32_t *op, uint32_t *offset, uint32_t *data) {
  uint8_t buf[12];
  if (g_fd < 0 || !RecvAll(g_fd, buf, sizeof(buf)))
    return -1;
  *op     = Le32(buf);
  *offset = Le32(buf + 4);
  *data   = Le32(buf + 8);
  return 0;
}

void f2_host_socket_reply(uint32_t data) {
  const uint8_t buf[4] = {static_cast<uint8_t>(data), static_cast<uint8_t>(data >> 8),
                          static_cast<uint8_t>(data >> 16), static_cast<uint8_t>(data >> 24)};
  if (g_fd >= 0)
    SendAll(g_fd, buf, sizeof(buf));
}

void f2_host_socket_close() {
  if (g_fd >= 0)
    close(g_fd);
  if (g_listen_fd >= 0)
    close(g_listen_fd);
  g_fd = g_listen_fd = -1;
}

}  // extern "C"
