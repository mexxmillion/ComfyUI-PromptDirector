import { app } from "../../scripts/app.js";

const PROVIDER_FIELDS = {
    "local llama": ["profile", "local_llama_thinking"],
    openrouter: ["openrouter_api_key", "openrouter_model", "reasoning_effort"],
    "custom endpoint": ["api_base", "local_model_slug", "local_api_key", "disable_endpoint_thinking"],
};
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

function syncFields(node) {
    const providerWidget = widget(node, "prompt_provider");
    if (!providerWidget) return;
    const provider = LEGACY_PROVIDERS[providerWidget.value] || providerWidget.value;
    if (provider !== providerWidget.value) providerWidget.value = provider;
    for (const [name, fields] of Object.entries(PROVIDER_FIELDS)) {
        for (const field of fields) showWidget(widget(node, field), provider === name);
    }
    showWidget(widget(node, "mode"), provider !== "none");
    showWidget(widget(node, "max_attempts"), provider !== "none");
    showWidget(widget(node, "reference_context"), false);
    showWidget(widget(node, "system_prompt_source"), false);
    showWidget(widget(node, "system_prompt_file"), provider !== "none" && widget(node, "mode")?.value === "Custom system prompt file");
    showWidget(widget(node, "custom_system_prompt"), provider !== "none" && widget(node, "mode")?.value === "Custom system prompt");
    showWidget(widget(node, "duration"), provider !== "none" && widget(node, "mode")?.value?.startsWith("MiniMax H3"));
    node.setSize?.(node.computeSize());
    app.graph?.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "prompt-director.provider-fields",
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
