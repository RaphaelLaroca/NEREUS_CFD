// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef SUITE_MAIN_H
#define SUITE_MAIN_H

// The driver every test suite shares. Each case is a former stand-alone harness: it runs in a child process of its
// own, so no global state (the grids, the scene, the method registry, the solvers' buffers) carries from one case to
// the next. Every case runs in $SUITE_WORK (tests/work/run, which tests/run_all.sh fills with the pressure configuration
// and a link to the fixtures), so nothing is written outside tests/. An argument "@WORKDIR" is replaced by a fresh
// directory made inside it, removed afterwards. The `scratch` field is kept for the case tables and no longer used.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

struct SuiteCase
{
    const char *label;
    int (*run)(int, char **);
    std::vector<const char *> args;
    bool scratch;
};

// Runs every case (or, with an argument, the cases whose label contains it) and prints one line per case at the end.
inline int RunSuite(const char *suite, const SuiteCase *cases, size_t count, int argc, char **argv)
{
    const char *filter = argc > 1 ? argv[1] : nullptr;
    const char *workRoot = std::getenv("SUITE_WORK");
    std::vector<std::string> failed;
    int ran = 0;
    for (size_t c = 0; c < count; c++)
    {
        const SuiteCase &k = cases[c];
        if (filter && !std::strstr(k.label, filter)) continue;
        ran++;
        std::printf("\n######## %s: %s ########\n", suite, k.label);
        std::fflush(stdout);
        std::string work;
        std::vector<std::string> args{k.label};
        for (const char *a : k.args)
        {
            if (std::strcmp(a, "@WORKDIR") == 0)
            {
                std::string tmpl = std::string(workRoot ? workRoot : ".") + "/case_XXXXXX";
                work = mkdtemp(tmpl.data()) ? tmpl : std::string(workRoot ? workRoot : ".");
                args.push_back(work);
            }
            else
                args.push_back(a);
        }
        const pid_t pid = fork();
        if (pid == 0)
        {
            if (workRoot && chdir(workRoot) != 0)
                std::perror("chdir");
            std::vector<char *> av;
            for (std::string &a : args) av.push_back(a.data());
            av.push_back(nullptr);
            const int rc = k.run((int)args.size(), av.data());
            std::fflush(stdout);
            std::fflush(stderr);
            _exit(rc);
        }
        int status = 0;
        waitpid(pid, &status, 0);
        if (!work.empty()) std::filesystem::remove_all(work);
        if (!(WIFEXITED(status) && WEXITSTATUS(status) == 0))
            failed.push_back(k.label);
    }
    std::printf("\n======== %s: %d case(s), %zu failed ========\n", suite, ran, failed.size());
    for (const std::string &f : failed) std::printf("  FAILED: %s\n", f.c_str());
    return failed.empty() ? 0 : 1;
}

#endif
