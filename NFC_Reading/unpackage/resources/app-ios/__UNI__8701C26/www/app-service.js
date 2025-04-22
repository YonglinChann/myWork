(function(vue) {
  "use strict";
  class NFCResultImpl {
    constructor(messages) {
      this.messages = messages;
    }
  }
  class NFCMessageImpl {
    constructor(records) {
      this.records = records;
    }
  }
  class NFCRecordImpl {
    constructor(typeNameFormat = null, type = null, identifier = null, payload = null) {
      this.typeNameFormat = typeNameFormat;
      this.type = type;
      this.identifier = identifier;
      this.payload = payload;
    }
  }
  class NFCAdapter {
    // 检查NFC是否可用
    static isAvailable() {
      return true;
    }
    // 开始扫描NFC标签
    static startScan(successCallback, errorCallback) {
      this.successCallback = successCallback;
      this.errorCallback = errorCallback;
      try {
        uni.showToast({
          title: "请将NFC标签靠近手机背面",
          icon: "none",
          duration: 2e3
        });
        setTimeout(() => {
          if (this.successCallback != null) {
            const record = new NFCRecordImpl(1, "Text", "id:1", "这是一条模拟的NFC标签信息");
            const message = new NFCMessageImpl([record]);
            const mockResult = new NFCResultImpl([message]);
            this.successCallback(mockResult);
          }
        }, 2e3);
      } catch (error) {
        if (this.errorCallback != null) {
          this.errorCallback("NFC扫描失败: " + String(error));
        }
      }
    }
    // 停止扫描
    static stopScan() {
      this.successCallback = null;
      this.errorCallback = null;
    }
  }
  NFCAdapter.successCallback = null;
  NFCAdapter.errorCallback = null;
  function isAvailable() {
    return NFCAdapter.isAvailable();
  }
  const _sfc_main$1 = vue.defineComponent({
    data() {
      return {
        scanResult: "",
        isScanning: false
        // 是否正在扫描
      };
    },
    onLoad() {
      this.checkNFCAvailability();
    },
    methods: {
      // 检查NFC是否可用
      checkNFCAvailability() {
        try {
          const isNFCAvailable = isAvailable();
          if (!isNFCAvailable) {
            uni.showToast({
              title: "您的设备不支持NFC功能或NFC功能未开启",
              icon: "none",
              duration: 3e3
            });
          } else {
            uni.__log__("log", "at pages/index/index.uvue:45", "NFC功能可用");
          }
        } catch (error) {
          uni.showToast({
            title: "检测NFC可用性失败: " + String(error),
            icon: "none",
            duration: 3e3
          });
          uni.__log__("error", "at pages/index/index.uvue:53", "检测NFC可用性失败:", error);
        }
      },
      // 开始NFC扫描
      startNFCScan() {
        if (this.isScanning) {
          uni.showToast({
            title: "正在扫描中，请稍候",
            icon: "none"
          });
          return null;
        }
        try {
          this.scanResult = "";
          this.isScanning = true;
          NFCAdapter.startScan(
            // 成功回调
            (result) => {
              this.isScanning = false;
              if (result && result.messages && result.messages.length > 0) {
                let scanResult = "";
                result.messages.forEach((message, index) => {
                  scanResult += "消息 ".concat(index + 1, ":\n");
                  if (message.records && message.records.length > 0) {
                    message.records.forEach((record, recordIndex) => {
                      scanResult += "记录 ".concat(recordIndex + 1, ":\n");
                      if (record.typeNameFormat !== void 0) {
                        scanResult += "类型格式: ".concat(this.getTypeNameFormatString(record.typeNameFormat), "\n");
                      }
                      if (record.type) {
                        scanResult += "类型: ".concat(record.type, "\n");
                      }
                      if (record.identifier) {
                        scanResult += "标识符: ".concat(record.identifier, "\n");
                      }
                      if (record.payload) {
                        scanResult += "载荷: ".concat(record.payload, "\n");
                      }
                      scanResult += "\n";
                    });
                  } else {
                    scanResult += "没有记录\n";
                  }
                  scanResult += "\n";
                });
                this.scanResult = scanResult;
              } else {
                this.scanResult = "未检测到NDEF数据";
              }
            },
            // 错误回调
            (error) => {
              this.isScanning = false;
              this.scanResult = "扫描错误: " + error;
              uni.__log__("error", "at pages/index/index.uvue:126", "NFC扫描错误:", error);
            }
          );
        } catch (error) {
          this.isScanning = false;
          uni.showToast({
            title: "启动NFC扫描失败: " + String(error),
            icon: "none"
          });
          uni.__log__("error", "at pages/index/index.uvue:135", "启动NFC扫描失败:", error);
        }
      },
      // 辅助方法: 将TNF值转换为可读字符串
      getTypeNameFormatString(tnf = null) {
        const tnfMap = new UTSJSONObject({
          0: "空",
          1: "NFC论坛类型",
          2: "媒体类型",
          3: "绝对URI",
          4: "NFC论坛外部类型",
          5: "未知",
          6: "不变",
          7: "保留"
        });
        return tnfMap[tnf] || "未知类型";
      }
    }
  });
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
    const _component_app = vue.resolveComponent("app", true);
    return vue.openBlock(), vue.createBlock(_component_app, null, {
      default: vue.withCtx(() => [
        vue.createVNode(_component_page)
      ]),
      _: 1
    });
  }
  const App = /* @__PURE__ */ _export_sfc(_sfc_main, [["render", _sfc_render], ["styles", [_style_0]]]);
  const __global__ = typeof globalThis === "undefined" ? Function("return this")() : globalThis;
  __global__.__uniX = true;
  function createApp() {
    const app = vue.createSSRApp(App);
    return {
      app
    };
  }
  createApp().app.mount("#app");
})(Vue);
