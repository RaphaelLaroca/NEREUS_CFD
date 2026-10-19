// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef SIMULATION_MANAGER_UI_H
#define SIMULATION_MANAGER_UI_H

// The optional user interface: a window with the setup screen, then the viewer. A program that wants it calls
//
//     std::string config;
//     if (!InitializeUI(config)) { DisposeUI(); return 0; }   // window, setup screen; false if it was closed
//     do
//     {
//         SimulationManager::InitializeSimulation(config);
//         do SimulationManager::StepSimulation(); while (UpdateUI());
//         SimulationManager::DisposeSimulation();
//     } while (RestartUI(config));                             // Restart pressed: the setup screen again
//     DisposeUI();
//
// Everything ImGui/ImPlot/GLFW is behind these four (src/UI/SimulationManagerUI.cpp, GUI builds only).

#include <string>

// Opens the window and runs the setup screen. True with `configFile` set when Start was pressed.
bool InitializeUI(std::string &configFile);
// Draws one frame of the viewer. False once the window is closing.
bool UpdateUI();
// After UpdateUI returned false: true when Restart was pressed and then Start on the setup screen, `configFile` the
// file to run; false when the window is closing. Call it after DisposeSimulation.
bool RestartUI(std::string &configFile);
// Closes the window and releases the UI. Safe to call after a failed InitializeUI.
void DisposeUI();

#endif
