(function(vue) {
  "use strict";
  const _sfc_main$1 = vue.defineComponent(new UTSJSONObject({
    data() {
      return {
        scanResult: "",
        nfcAdapter: null
        // NFC适配器实例
      };
    },
    onLoad() {
      this.checkNFCAvailability();
    },
    methods: new UTSJSONObject({
      // 检查NFC是否可用
      checkNFCAvailability() {
        try {
          this.nfcAdapter = uni.getNFCAdapter();
          const isNFCAvailable = this.nfcAdapter.isNFCAvailable();
          if (!isNFCAvailable) {
            uni.showToast({
              title: "您的设备不支持NFC功能或NFC功能未开启",
              icon: "none",
              duration: 3e3
            });
          }
        } catch (error) {
          uni.showToast({
            title: "初始化NFC适配器失败: " + error.message,
            icon: "none",
            duration: 3e3
          });
          uni.__log__("error", "at pages/index/index.uvue:51", "初始化NFC适配器失败:", error);
        }
      },
      // 开始NFC扫描
      startNFCScan() {
        if (!this.nfcAdapter) {
          uni.showToast({
            title: "NFC适配器未初始化",
            icon: "none"
          });
          return null;
        }
        try {
          this.scanResult = "";
          uni.showToast({
            title: "请将NFC标签靠近手机背面",
            icon: "none",
            duration: 3e3
          });
          const session = this.nfcAdapter.getNdefTagSession();
          session.onNdefMessage((res = null) => {
            if (res && res.messages && res.messages.length > 0) {
              let result = "";
              res.messages.forEach((message = null, index = null) => {
                result += "消息 ".concat(index + 1, ":\n");
                if (message.records && message.records.length > 0) {
                  message.records.forEach((record = null, recordIndex = null) => {
                    result += "记录 ".concat(recordIndex + 1, ":\n");
                    result += "TNF: ".concat(record.tnf, "\n");
                    result += "类型: ".concat(record.type, "\n");
                    if (record.payload) {
                      try {
                        const decoder = new TextDecoder("utf-8");
                        const text = decoder.decode(new Uint8Array(record.payload));
                        result += "载荷: ".concat(text, "\n");
                      } catch (e) {
                        result += "载荷: [二进制数据]\n";
                      }
                    }
                    result += "\n";
                  });
                } else {
                  result += "没有记录\n";
                }
                result += "\n";
              });
              this.scanResult = result;
            } else {
              this.scanResult = "未检测到NDEF数据";
            }
            session.close();
          });
          session.onError((error = null) => {
            this.scanResult = "扫描错误: " + error.errMsg;
            session.close();
          });
          session.start();
        } catch (error) {
          uni.showToast({
            title: "启动NFC扫描失败: " + error.message,
            icon: "none"
          });
          uni.__log__("error", "at pages/index/index.uvue:136", "启动NFC扫描失败:", error);
        }
      }
    })
  }));
  const _style_0$1 = { "container": { "": { "flex": 1, "paddingTop": 20, "paddingRight": 20, "paddingBottom": 20, "paddingLeft": 20, "backgroundColor": "#f8f8f8" } }, "title": { "": { "fontSize": 24, "fontWeight": "bold", "textAlign": "center", "marginBottom": 40, "color": "#333333" } }, "scan-btn": { "": { "backgroundColor": "#007aff", "color": "#ffffff", "borderTopLeftRadius": 8, "borderTopRightRadius": 8, "borderBottomRightRadius": 8, "borderBottomLeftRadius": 8, "paddingTop": 15, "paddingRight": 0, "paddingBottom": 15, "paddingLeft": 0, "fontSize": 18, "marginBottom": 30 } }, "result-area": { "": { "borderTopLeftRadius": 8, "borderTopRightRadius": 8, "borderBottomRightRadius": 8, "borderBottomLeftRadius": 8, "backgroundColor": "#ffffff", "paddingTop": 15, "paddingRight": 15, "paddingBottom": 15, "paddingLeft": 15, "borderTopWidth": 1, "borderRightWidth": 1, "borderBottomWidth": 1, "borderLeftWidth": 1, "borderTopStyle": "solid", "borderRightStyle": "solid", "borderBottomStyle": "solid", "borderLeftStyle": "solid", "borderTopColor": "#e0e0e0", "borderRightColor": "#e0e0e0", "borderBottomColor": "#e0e0e0", "borderLeftColor": "#e0e0e0", "marginTop": 20 } }, "result-title": { "": { "fontSize": 18, "fontWeight": "bold", "marginBottom": 10, "color": "#333333" } }, "result-content": { "": { "height": 300, "paddingTop": 10, "paddingRight": 10, "paddingBottom": 10, "paddingLeft": 10, "backgroundColor": "#f0f0f0", "borderTopLeftRadius": 5, "borderTopRightRadius": 5, "borderBottomRightRadius": 5, "borderBottomLeftRadius": 5 } } };
  const _export_sfc = (sfc, props) => {
    const target = sfc.__vccOpts || sfc;
    for (const [key, val] of props) {
      target[key] = val;
    }
    return target;
  };
  function _sfc_render$1(_ctx, _cache, $props, $setup, $data, $options) {
    return vue.openBlock(), vue.createElementBlock("view", { class: "container" }, [
      vue.createElementVNode("text", { class: "title" }, "NFC 标签读取应用"),
      vue.createElementVNode("button", {
        class: "scan-btn",
        onClick: _cache[0] || (_cache[0] = (...args) => $options.startNFCScan && $options.startNFCScan(...args))
      }, "开始扫描 NFC 标签"),
      $data.scanResult ? (vue.openBlock(), vue.createElementBlock("view", {
        key: 0,
        class: "result-area"
      }, [
        vue.createElementVNode("text", { class: "result-title" }, "扫描结果:"),
        vue.createElementVNode("scroll-view", { class: "result-content" }, [
          vue.createElementVNode("text", null, vue.toDisplayString($data.scanResult), 1)
        ])
      ])) : vue.createCommentVNode("", true)
    ]);
  }
  const PagesIndexIndex = /* @__PURE__ */ _export_sfc(_sfc_main$1, [["render", _sfc_render$1], ["styles", [_style_0$1]]]);
  __definePage("pages/index/index", PagesIndexIndex);
  const _sfc_main = vue.defineComponent(new UTSJSONObject({
    onLaunch: function() {
      uni.__log__("log", "at App.uvue:10", "App Launch");
    },
    onShow: function() {
      uni.__log__("log", "at App.uvue:13", "App Show");
    },
    onHide: function() {
      uni.__log__("log", "at App.uvue:16", "App Hide");
    }
  }));
  const _style_0 = {};
  function _sfc_render(_ctx, _cache, $props, $setup, $data, $options) {
    const _component_page = vue.resolveComponent("page");
    return vue.openBlock(), vue.createElementBlock("view", null, [
      vue.createVNode(_component_page)
    ]);
  }
  const App = /* @__PURE__ */ _export_sfc(_sfc_main, [["render", _sfc_render], ["styles", [_style_0]]]);
  const __global__ = typeof globalThis === "undefined" ? Function("return this")() : globalThis;
  __global__.__uniX = true;
  const app = vue.createApp(App);
  app.mount("#app");
  vue.createApp().app.mount("#app");
})(Vue);
