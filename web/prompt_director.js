import { app } from "../../scripts/app.js";

const PROVIDER_FIELDS = {
    "local llama": ["profile"],
    openrouter: ["openrouter_model"],
    "custom endpoint": ["api_base"],
};
const SETTINGS_FIELDS = [
    "openrouter_api_key", "reasoning_effort", "local_model_slug", "local_api_key",
    "disable_endpoint_thinking", "local_llama_thinking", "max_attempts", "fail_on_error",
    "system_prompt_source", "system_prompt_file", "custom_system_prompt",
];
const LEGACY_PROVIDERS = {
    local: "local llama",
    "managed llama-server": "local llama",
    "local endpoint": "custom endpoint",
};

function widget(node, name) {
    return node.widgets?.find((item) => item.name === name);
}

function showWidget(item, visible) {
    if (!item) return;
    if (item._directorType === undefined) {
        item._directorType = item.type;
        item._directorDraw = item.draw;
        item._directorComputeSize = item.computeSize;
    }
    item.type = visible ? item._directorType : "hidden";
    item.hidden = !visible;
    item.computeSize = visible ? item._directorComputeSize : () => [0, 0];
    item.draw = visible ? item._directorDraw : () => {};
    if (!visible) item.last_y = undefined;
}

function providerValue(node) {
    const providerWidget = widget(node, "prompt_provider");
    if (!providerWidget) return "none";
    const provider = LEGACY_PROVIDERS[providerWidget.value] || providerWidget.value;
    if (provider !== providerWidget.value) providerWidget.value = provider;
    return provider;
}

function syncFields(node) {
    const provider = providerValue(node);
    for (const [name, fields] of Object.entries(PROVIDER_FIELDS)) {
        for (const field of fields) showWidget(widget(node, field), provider === name);
    }
    for (const field of SETTINGS_FIELDS) showWidget(widget(node, field), false);
    showWidget(widget(node, "reference_context"), false);
    showWidget(widget(node, "mode"), provider !== "none");
    showWidget(widget(node, "creativity"), provider !== "none");
    showWidget(widget(node, "width"), provider !== "none");
    showWidget(widget(node, "height"), provider !== "none");
    showWidget(widget(node, "duration"), provider !== "none" && widget(node, "mode")?.value?.startsWith("MiniMax H3"));
    node.setSize?.(node.computeSize());
    app.graph?.setDirtyCanvas(true, true);
}

function addStyles() {
    if (document.getElementById("prompt-director-settings-style")) return;
    const style = document.createElement("style");
    style.id = "prompt-director-settings-style";
    style.textContent = `
        .prompt-director-settings { width: min(620px, calc(100vw - 40px)); color: var(--fg-color, #ddd); background: var(--comfy-menu-bg, #252525); border: 1px solid #555; border-radius: 10px; padding: 0; }
        .prompt-director-settings::backdrop { background: rgba(0, 0, 0, .65); }
        .prompt-director-settings form { display: grid; gap: 14px; padding: 20px; }
        .prompt-director-settings h2 { margin: 0; font-size: 18px; }
        .prompt-director-settings h3 { margin: 8px 0 0; font-size: 14px; }
        .prompt-director-settings label { display: grid; gap: 5px; font-size: 12px; color: #bbb; }
        .prompt-director-settings input, .prompt-director-settings select, .prompt-director-settings textarea { box-sizing: border-box; width: 100%; color: #eee; background: #171717; border: 1px solid #555; border-radius: 5px; padding: 8px; }
        .prompt-director-settings textarea { min-height: 210px; resize: vertical; font: 12px/1.45 monospace; }
        .prompt-director-settings .director-check { display: flex; align-items: center; gap: 8px; }
        .prompt-director-settings .director-check input { width: auto; }
        .prompt-director-settings .director-note { margin: -7px 0 0; color: #999; font-size: 11px; }
        .prompt-director-settings .director-actions { display: flex; justify-content: flex-end; gap: 8px; }
        .prompt-director-settings button { color: #eee; background: #3a3a3a; border: 1px solid #666; border-radius: 5px; padding: 7px 12px; cursor: pointer; }
        .prompt-director-settings button[type=submit] { background: #245b85; }
    `;
    document.head.appendChild(style);
}

function field(labelText, control) {
    const label = document.createElement("label");
    const caption = document.createElement("span");
    caption.textContent = labelText;
    label.append(caption, control);
    return label;
}

function textInput(value, type = "text") {
    const input = document.createElement("input");
    input.type = type;
    input.value = value ?? "";
    return input;
}

function numberInput(value, min, max) {
    const input = textInput(value, "number");
    input.min = String(min);
    input.max = String(max);
    input.step = "1";
    return input;
}

function checkbox(labelText, value) {
    const label = document.createElement("label");
    label.className = "director-check";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = Boolean(value);
    const caption = document.createElement("span");
    caption.textContent = labelText;
    label.append(input, caption);
    return { label, input };
}

function selectInput(values, value) {
    const select = document.createElement("select");
    for (const optionValue of values) {
        const option = document.createElement("option");
        option.value = optionValue;
        option.textContent = optionValue;
        option.selected = optionValue === value;
        select.appendChild(option);
    }
    return select;
}

function setValue(node, name, value) {
    const item = widget(node, name);
    if (!item) return;
    item.value = value;
    item.callback?.(value);
}

function openSettings(node) {
    addStyles();
    const provider = providerValue(node);
    const dialog = document.createElement("dialog");
    dialog.className = "prompt-director-settings";
    const form = document.createElement("form");
    form.method = "dialog";
    const title = document.createElement("h2");
    title.textContent = "Prompt Director settings";
    form.appendChild(title);

    const values = {};
    if (provider === "local llama") {
        const heading = document.createElement("h3");
        heading.textContent = "Local model";
        values.local_llama_thinking = checkbox("Allow model thinking", widget(node, "local_llama_thinking")?.value);
        form.append(heading, values.local_llama_thinking.label);
    } else if (provider === "openrouter") {
        const heading = document.createElement("h3");
        heading.textContent = "OpenRouter";
        values.openrouter_api_key = textInput(widget(node, "openrouter_api_key")?.value, "password");
        values.reasoning_effort = selectInput(["none", "low", "medium", "high"], widget(node, "reasoning_effort")?.value);
        form.append(heading, field("API key (blank uses environment)", values.openrouter_api_key), field("Reasoning", values.reasoning_effort));
    } else if (provider === "custom endpoint") {
        const heading = document.createElement("h3");
        heading.textContent = "Endpoint details";
        values.local_model_slug = textInput(widget(node, "local_model_slug")?.value);
        values.local_api_key = textInput(widget(node, "local_api_key")?.value, "password");
        values.disable_endpoint_thinking = checkbox("Disable endpoint thinking", widget(node, "disable_endpoint_thinking")?.value);
        form.append(heading, field("Model ID", values.local_model_slug), field("API key (optional)", values.local_api_key), values.disable_endpoint_thinking.label);
    }

    const behaviorHeading = document.createElement("h3");
    behaviorHeading.textContent = "Behavior";
    values.max_attempts = numberInput(widget(node, "max_attempts")?.value, 1, 4);
    values.fail_on_error = checkbox("Stop workflow if enhancement fails", widget(node, "fail_on_error")?.value);
    form.append(behaviorHeading, field("Attempts", values.max_attempts), values.fail_on_error.label);

    const systemHeading = document.createElement("h3");
    systemHeading.textContent = "System prompt override";
    const textarea = document.createElement("textarea");
    textarea.value = widget(node, "custom_system_prompt")?.value || "";
    textarea.placeholder = "Leave blank to use the selected output type's built-in system prompt.";
    values.custom_system_prompt = textarea;
    const note = document.createElement("p");
    note.className = "director-note";
    note.textContent = "This replaces the built-in system prompt only for this node.";
    const loadButton = document.createElement("button");
    loadButton.type = "button";
    loadButton.textContent = "Load .md / .txt";
    loadButton.onclick = () => {
        const picker = document.createElement("input");
        picker.type = "file";
        picker.accept = ".md,.txt,text/markdown,text/plain";
        picker.onchange = async () => {
            const file = picker.files?.[0];
            if (file) textarea.value = await file.text();
        };
        picker.click();
    };
    const clearButton = document.createElement("button");
    clearButton.type = "button";
    clearButton.textContent = "Use built-in";
    clearButton.onclick = () => { textarea.value = ""; };
    const promptActions = document.createElement("div");
    promptActions.className = "director-actions";
    promptActions.append(loadButton, clearButton);
    form.append(systemHeading, textarea, note, promptActions);

    const cancel = document.createElement("button");
    cancel.type = "button";
    cancel.textContent = "Cancel";
    cancel.onclick = () => dialog.close();
    const apply = document.createElement("button");
    apply.type = "submit";
    apply.textContent = "Apply";
    const actions = document.createElement("div");
    actions.className = "director-actions";
    actions.append(cancel, apply);
    form.appendChild(actions);

    form.onsubmit = () => {
        for (const [name, control] of Object.entries(values)) {
            const input = control.input || control;
            const value = input.type === "checkbox" ? input.checked : input.type === "number" ? Number(input.value) : input.value;
            setValue(node, name, value);
        }
        setValue(node, "system_prompt_source", "built-in for output target");
        setValue(node, "system_prompt_file", "");
        app.graph?.setDirtyCanvas(true, true);
    };
    dialog.addEventListener("close", () => dialog.remove());
    dialog.appendChild(form);
    document.body.appendChild(dialog);
    dialog.showModal();
}

app.registerExtension({
    name: "prompt-director.simple-fields",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "PromptDirector") return;
        const originalCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalCreated?.apply(this, arguments);
            for (const name of ["prompt_provider", "mode"]) {
                const item = widget(this, name);
                if (!item) continue;
                const originalCallback = item.callback;
                item.callback = (...args) => {
                    const value = originalCallback?.apply(item, args);
                    syncFields(this);
                    return value;
                };
            }
            const settings = this.addWidget("button", "⚙ Prompt settings", null, () => openSettings(this));
            settings.serialize = false;
            syncFields(this);
            return result;
        };
        const originalConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = originalConfigure?.apply(this, arguments);
            syncFields(this);
            return result;
        };
    },
});
