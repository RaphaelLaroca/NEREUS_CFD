// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef SETTINGS_PANEL_H
#define SETTINGS_PANEL_H

// The default widgets for a SettingField table (slider, checkbox, input box, combo, help as tooltip).
// GUI builds only.

#include <algorithm>
#include "imgui.h"
#include "Definitions.h"

// Status text colours that read on either theme: the dark theme keeps the bright tones, the light
// one (picked off the window background) gets darker ones, since yellow and green on paper vanish.
enum class UiTone { Warn, Error, Good, Note };
inline ImVec4 ToneColour(UiTone t)
{
    const ImVec4 bg = ImGui::GetStyle().Colors[ImGuiCol_WindowBg];
    const bool light = 0.299f * bg.x + 0.587f * bg.y + 0.114f * bg.z > 0.5f;
    switch (t)
    {
        case UiTone::Warn:  return light ? ImVec4(0.60f, 0.36f, 0.00f, 1.0f) : ImVec4(1.00f, 0.82f, 0.40f, 1.0f);
        case UiTone::Error: return light ? ImVec4(0.75f, 0.12f, 0.10f, 1.0f) : ImVec4(1.00f, 0.45f, 0.40f, 1.0f);
        case UiTone::Good:  return light ? ImVec4(0.08f, 0.45f, 0.14f, 1.0f) : ImVec4(0.55f, 0.90f, 0.55f, 1.0f);
        case UiTone::Note:  return light ? ImVec4(0.70f, 0.30f, 0.00f, 1.0f) : ImVec4(1.00f, 0.75f, 0.30f, 1.0f);
    }
    return ImVec4(1, 1, 1, 1);
}

// ---- form rows: the label on the left, the box in a column shared by every row, the unit after the value ----

inline float FormEm() { return ImGui::CalcTextSize("M").x; }
inline float FormNumberWidth() { return FormEm() * 14.0f; }   // a number box
inline float FormWideWidth() { return FormEm() * 40.0f; }     // an expression box
inline float FormAxisWidth() { return FormEm() * 9.0f; }      // one component of a point

// The window x of the box column: one indent plus room for a 22-character label, so the boxes line up at every
// indent. A longer label pushes its own box right.
inline float FormColumn() { return ImGui::GetStyle().IndentSpacing + FormEm() * 23.0f; }

inline void FormTooltip(const char *help)
{
    if (help && help[0] && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
        ImGui::SetTooltip("%s", help);
}

// The row's label; the next item starts at the box column, or after the label if it runs past it.
inline void FormLabel(const char *label, const char *help = nullptr)
{
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted(label);
    FormTooltip(help);
    const float labelEnd = ImGui::GetItemRectMax().x - ImGui::GetWindowPos().x + ImGui::GetScrollX();
    ImGui::SameLine((std::max)(FormColumn(), labelEnd + ImGui::GetStyle().ItemSpacing.x));
}

// The unit after a value, dimmed. Nothing for nullptr or "".
inline void FormUnit(const char *unit)
{
    if (!unit || !unit[0]) return;
    ImGui::SameLine();
    ImGui::TextDisabled("%s", unit);
}

inline bool FormDouble(const char *label, double *v, const char *unit = nullptr, const char *fmt = "%.6g",
                       const char *help = nullptr)
{
    FormLabel(label, help);
    ImGui::PushID(label);
    ImGui::SetNextItemWidth(FormNumberWidth());
    const bool c = ImGui::InputDouble("##v", v, 0.0, 0.0, fmt);
    FormTooltip(help);
    ImGui::PopID();
    FormUnit(unit);
    return c;
}

inline bool FormInt(const char *label, int *v, const char *unit = nullptr, const char *help = nullptr, int step = 1,
                    int stepFast = 100)
{
    FormLabel(label, help);
    ImGui::PushID(label);
    // Wider by the two step buttons, so the number itself gets a number box.
    const ImGuiStyle &st = ImGui::GetStyle();
    ImGui::SetNextItemWidth(FormNumberWidth() + 2.0f * (ImGui::GetFrameHeight() + st.ItemInnerSpacing.x));
    const bool c = ImGui::InputInt("##v", v, step, stepFast);
    FormTooltip(help);
    ImGui::PopID();
    FormUnit(unit);
    return c;
}

inline bool FormSliderInt(const char *label, int *v, int lo, int hi, const char *unit = nullptr,
                          const char *help = nullptr)
{
    FormLabel(label, help);
    ImGui::PushID(label);
    ImGui::SetNextItemWidth(FormNumberWidth());
    const bool c = ImGui::SliderInt("##v", v, lo, hi);
    FormTooltip(help);
    ImGui::PopID();
    FormUnit(unit);
    return c;
}

inline bool FormCombo(const char *label, int *v, const char *const *items, int count, const char *help = nullptr,
                      float width = 0.0f)
{
    FormLabel(label, help);
    ImGui::PushID(label);
    ImGui::SetNextItemWidth(width > 0.0f ? width : FormNumberWidth());
    const bool c = ImGui::Combo("##v", v, items, count);
    FormTooltip(help);
    ImGui::PopID();
    return c;
}

// Items as one string, each ended by '\0'.
inline bool FormCombo(const char *label, int *v, const char *itemsZeroSeparated, const char *help = nullptr,
                      float width = 0.0f)
{
    FormLabel(label, help);
    ImGui::PushID(label);
    ImGui::SetNextItemWidth(width > 0.0f ? width : FormNumberWidth());
    const bool c = ImGui::Combo("##v", v, itemsZeroSeparated);
    FormTooltip(help);
    ImGui::PopID();
    return c;
}

// A point or a vector: each component in its own box, after the name of its axis.
inline bool FormVec(const char *label, double *v, int n, const char *unit = nullptr, const char *fmt = "%.5g",
                    const char *help = nullptr)
{
    static const char *axis[3] = {"x", "y", "z"};
    FormLabel(label, help);
    ImGui::PushID(label);
    bool c = false;
    for (int a = 0; a < n; a++)
    {
        if (a > 0) ImGui::SameLine();
        ImGui::TextDisabled("%s", axis[a]);
        ImGui::SameLine(0.0f, FormEm() * 0.5f);
        ImGui::SetNextItemWidth(FormAxisWidth());
        ImGui::PushID(a);
        c |= ImGui::InputDouble("##v", &v[a], 0.0, 0.0, fmt);
        FormTooltip(help);
        ImGui::PopID();
    }
    ImGui::PopID();
    FormUnit(unit);
    return c;
}

// One field. True when the user changed it.
inline bool DrawSettingField(const SettingField &f, void *base)
{
    bool changed = false;
    ImGui::PushID(f.key);
    switch (f.kind)
    {
        case SETTING_INT:
        {
            int v = (int)GetSettingValue(base, f);
            if (f.minValue > SETTING_NO_MIN && f.maxValue < SETTING_NO_MAX)
                changed = FormSliderInt(f.label, &v, (int)f.minValue, (int)f.maxValue, f.unit, f.help);
            else
            {
                changed = FormInt(f.label, &v, f.unit, f.help);
                if (f.minValue > SETTING_NO_MIN) v = (std::max)(v, (int)f.minValue);
                if (f.maxValue < SETTING_NO_MAX) v = (std::min)(v, (int)f.maxValue);
            }
            if (changed) SetSettingValue(base, f, v);
            break;
        }
        case SETTING_DOUBLE:
        {
            double v = GetSettingValue(base, f);
            changed = FormDouble(f.label, &v, f.unit, "%.6g", f.help);
            if (changed)
            {
                v = (std::max)(f.minValue, (std::min)(f.maxValue, v));
                SetSettingValue(base, f, v);
            }
            break;
        }
        case SETTING_BOOL:
        {
            bool v = GetSettingValue(base, f) != 0.0;
            changed = ImGui::Checkbox(f.label, &v);
            FormTooltip(f.help);
            if (changed) SetSettingValue(base, f, v ? 1.0 : 0.0);
            break;
        }
        case SETTING_ENUM:
        {
            int v = (int)GetSettingValue(base, f);
            if (v < 0 || v >= f.enumCount) v = (int)f.defaultValue;
            changed = FormCombo(f.label, &v, f.enumLabels, f.enumCount, f.help, FormEm() * 24.0f);
            if (changed) SetSettingValue(base, f, v);
            break;
        }
    }
    ImGui::PopID();
    return changed;
}

// One field by key, for a custom panel that interleaves fields with its own readouts.
inline bool DrawSetting(const SettingField *fields, int n, void *base, const char *key)
{
    const SettingField *f = FindSettingField(fields, n, key);
    return f ? DrawSettingField(*f, base) : false;
}

// The whole table, in order: what a method without a custom panel gets.
inline bool DrawSettingFields(const SettingField *fields, int n, void *base)
{
    bool changed = false;
    for (int i = 0; i < n; i++)
        changed |= DrawSettingField(fields[i], base);
    return changed;
}

#endif
