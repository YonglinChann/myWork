(function(vue) {
  "use strict";
  function initRuntimeSocket(hosts, port, id) {
    if (hosts == "" || port == "" || id == "")
      return Promise.resolve(null);
    return hosts.split(",").reduce((promise, host) => {
      return promise.then((socket) => {
        if (socket != null)
          return Promise.resolve(socket);
        return tryConnectSocket(host, port, id);
      });
    }, Promise.resolve(null));
  }
  const SOCKET_TIMEOUT = 500;
  function tryConnectSocket(host, port, id) {
    return new Promise((resolve, reject) => {
      const socket = uni.connectSocket({
        url: "ws://".concat(host, ":").concat(port, "/").concat(id),
        fail() {
          resolve(null);
        }
      });
      const timer = setTimeout(() => {
        socket.close({
          code: 1006,
          reason: "connect timeout"
        });
        resolve(null);
      }, SOCKET_TIMEOUT);
      socket.onOpen((e) => {
        clearTimeout(timer);
        resolve(socket);
      });
      socket.onClose((e) => {
        clearTimeout(timer);
        resolve(null);
      });
      socket.onError((e) => {
        clearTimeout(timer);
        resolve(null);
      });
    });
  }
  function initRuntimeSocketService() {
    const hosts = "127.0.0.1,192.168.0.175,198.18.0.1,169.254.214.96";
    const port = "8090";
    const id = "app-ios__XTA26";
    let socketTask = null;
    __registerWebViewUniConsole(() => {
      return '!function(){"use strict";function e(e,t){try{return{type:e,args:n(t)}}catch(e){}return{type:e,args:[]}}function n(e){return e.map((function(e){return t(e)}))}function t(e,n){if(void 0===n&&(n=0),n>=7)return{type:"object",value:"[Maximum depth reached]"};switch(typeof e){case"string":return{type:"string",value:e};case"number":return function(e){return{type:"number",value:String(e)}}(e);case"boolean":return function(e){return{type:"boolean",value:String(e)}}(e);case"object":return function(e,n){if(null===e)return{type:"null"};if(function(e){return e.$&&r(e.$)}(e))return function(e,n){return{type:"object",className:"ComponentPublicInstance",value:{properties:Object.entries(e.$.type).map((function(e){return o(e[0],e[1],n+1)}))}}}(e,n);if(r(e))return function(e,n){return{type:"object",className:"ComponentInternalInstance",value:{properties:Object.entries(e.type).map((function(e){return o(e[0],e[1],n+1)}))}}}(e,n);if(function(e){return e.style&&null!=e.tagName&&null!=e.nodeName}(e))return function(e,n){return{type:"object",value:{properties:Object.entries(e).filter((function(e){var n=e[0];return["id","tagName","nodeName","dataset","offsetTop","offsetLeft","style"].includes(n)})).map((function(e){return o(e[0],e[1],n+1)}))}}}(e,n);if(function(e){return"function"==typeof e.getPropertyValue&&"function"==typeof e.setProperty&&e.$styles}(e))return function(e,n){return{type:"object",value:{properties:Object.entries(e.$styles).map((function(e){return o(e[0],e[1],n+1)}))}}}(e,n);if(Array.isArray(e))return{type:"object",subType:"array",value:{properties:e.map((function(e,r){return function(e,n,r){var o=t(e,r);return o.name="".concat(n),o}(e,r,n+1)}))}};if(e instanceof Set)return{type:"object",subType:"set",className:"Set",description:"Set(".concat(e.size,")"),value:{entries:Array.from(e).map((function(e){return function(e,n){return{value:t(e,n)}}(e,n+1)}))}};if(e instanceof Map)return{type:"object",subType:"map",className:"Map",description:"Map(".concat(e.size,")"),value:{entries:Array.from(e.entries()).map((function(e){return function(e,n){return{key:t(e[0],n),value:t(e[1],n)}}(e,n+1)}))}};if(e instanceof Promise)return{type:"object",subType:"promise",value:{properties:[]}};if(e instanceof RegExp)return{type:"object",subType:"regexp",value:String(e),className:"Regexp"};if(e instanceof Date)return{type:"object",subType:"date",value:String(e),className:"Date"};if(e instanceof Error)return{type:"object",subType:"error",value:e.message||String(e),className:e.name||"Error"};var a=void 0,i=e.constructor;i&&i.get$UTSMetadata$&&(a=i.get$UTSMetadata$().name);return{type:"object",className:a,value:{properties:Object.entries(e).map((function(e){return o(e[0],e[1],n+1)}))}}}(e,n);case"undefined":return{type:"undefined"};case"function":return function(e){return{type:"function",value:"function ".concat(e.name,"() {}")}}(e);case"symbol":return function(e){return{type:"symbol",value:e.description}}(e);case"bigint":return function(e){return{type:"bigint",value:String(e)}}(e)}}function r(e){return e.type&&null!=e.uid&&e.appContext}function o(e,n,r){var o=t(n,r);return o.name=e,o}"function"==typeof SuppressedError&&SuppressedError;var a=["log","warn","error","info","debug"],i=null,u=[],c={};function s(e){null!=i?i(JSON.stringify(Object.assign({type:"console",data:e},c))):u.push.apply(u,e)}var f=a.reduce((function(e,n){return e[n]=console[n].bind(console),e}),{}),p=/^\\s*at\\s+[\\w/./-]+:\\d+$/;function l(){function n(n){return function(){for(var t=[],r=0;r<arguments.length;r++)t[r]=arguments[r];var o=function(e,n,t){if(t||2===arguments.length)for(var r,o=0,a=n.length;o<a;o++)!r&&o in n||(r||(r=Array.prototype.slice.call(n,0,o)),r[o]=n[o]);return e.concat(r||Array.prototype.slice.call(n))}([],t,!0);if(o.length){var a=o[o.length-1];"string"==typeof a&&p.test(a)&&o.pop()}f[n].apply(f,o),s([e(n,t)])}}return function(){var e=console.log,n=Symbol();try{console.log=n}catch(e){return!1}var t=console.log===n;return console.log=e,t}()?(a.forEach((function(e){console[e]=n(e)})),function(){a.forEach((function(e){console[e]=f[e]}))}):function(){}}var y=null,g=new Set,d={};function v(e){if(null!=y){var n=e.map((function(e){var n=e&&"promise"in e&&"reason"in e,t=n?"UnhandledPromiseRejection: ":"";if(n&&(e=e.reason),e instanceof Error&&e.stack)return e.message&&!e.stack.includes(e.message)?"".concat(t).concat(e.message,"\\n").concat(e.stack):"".concat(t).concat(e.stack);if("object"==typeof e&&null!==e)try{return t+JSON.stringify(e)}catch(e){return t+String(e)}return t+String(e)})).filter(Boolean);n.length>0&&y(JSON.stringify(Object.assign({type:"error",data:n},d)))}else e.forEach((function(e){g.add(e)}))}function m(e){var n={type:"WEB_INVOKE_APPSERVICE",args:{data:{name:"console",arg:e}}};return window.__uniapp_x_postMessageToService?window.__uniapp_x_postMessageToService(n):window.__uniapp_x_.postMessageToService(JSON.stringify(n))}!function(){if(!window.__UNI_CONSOLE_WEBVIEW__){window.__UNI_CONSOLE_WEBVIEW__=!0;var e="[web-view]".concat(window.__UNI_PAGE_ROUTE__?"[".concat(window.__UNI_PAGE_ROUTE__,"]"):"");l(),function(e,n){if(void 0===n&&(n={}),i=e,Object.assign(c,n),null!=e&&u.length>0){var t=u.slice();u.length=0,s(t)}}((function(e){m(e)}),{channel:e}),function(e,n){if(void 0===n&&(n={}),y=e,Object.assign(d,n),null!=e&&g.size>0){var t=Array.from(g);g.clear(),v(t)}}((function(e){m(e)}),{channel:e}),window.addEventListener("error",(function(e){v([e.error])})),window.addEventListener("unhandledrejection",(function(e){v([e])}))}}()}();';
    }, (data) => {
      socketTask === null || socketTask === void 0 ? void 0 : socketTask.send({
        data
      });
    });
    return Promise.resolve().then(() => {
      return initRuntimeSocket(hosts, port, id).then((socket) => {
        if (socket == null) {
          return false;
        }
        socketTask = socket;
        return true;
      });
    }).catch(() => {
      return false;
    });
  }
  initRuntimeSocketService();
  class NFCAdapter {
    // 检查NFC是否可用
    static isAvailable() {
      return true;
    }
    // 开始扫描NFC标签
    static startScan(successCallback, errorCallback) {
      this.successCallback = successCallback;
      this.errorCallback = errorCallback;
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
          if (!isNFCAvailable)
            ;
          else {
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
          vue.createElementVNode(
            "text",
            null,
            vue.toDisplayString($data.scanResult),
            1
            /* TEXT */
          )
        ])
      ])) : vue.createCommentVNode("v-if", true)
    ]);
  }
  const PagesIndexIndex = /* @__PURE__ */ _export_sfc(_sfc_main$1, [["render", _sfc_render$1], ["styles", [_style_0$1]], ["__file", "/Users/chenyonglin/myCode/gitee/myWork/NFC_Reading/pages/index/index.uvue"]]);
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
      /* STABLE */
    });
  }
  const App = /* @__PURE__ */ _export_sfc(_sfc_main, [["render", _sfc_render], ["styles", [_style_0]], ["__file", "/Users/chenyonglin/myCode/gitee/myWork/NFC_Reading/App.uvue"]]);
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
//# sourceMappingURL=../../../cache/.app-ios/sourcemap/app-service.js.map
