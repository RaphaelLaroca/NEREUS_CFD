// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "Utils.h"
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

std::filesystem::path ExecutablePath(const char *argv0)
{
#ifdef _WIN32
    // argv[0] is in the ANSI code page; the wide API returns the path whatever characters its folders hold.
    std::wstring buf(32768, L'\0');
    const DWORD n = GetModuleFileNameW(nullptr, &buf[0], (DWORD)buf.size());
    if (n > 0 && n < buf.size())
    {
        buf.resize(n);
        return std::filesystem::path(buf);
    }
#endif
    if (argv0 == nullptr || argv0[0] == '\0') return std::filesystem::path();
    std::error_code ec;
    return std::filesystem::absolute(std::filesystem::path(argv0), ec);
}
