# NFC标签读取应用

这是一个使用 uni-app x 框架开发的iOS应用，用于读取NFC标签中的数据。

## 功能特点

- 点击按钮启动NFC标签扫描功能
- 扫描并读取NFC标签中的NDEF数据
- 显示读取结果，包括TNF、类型和载荷内容

## 开发环境要求

- HBuilderX 4.0+
- iOS 11.0+（支持NFC功能的设备）
- 开发者账号（用于iOS应用签名和发布）

## 注意事项

1. 在iOS上使用NFC功能需要:
   - 开发者账号
   - 在Apple开发者中心配置NFC相关权限
   - 添加相应的Entitlements

2. 应用需要在真机上运行测试，模拟器不支持NFC功能

3. 仅支持具有NFC功能的iPhone设备（iPhone 7及以上机型）

## 如何使用

1. 打开应用
2. 点击"开始扫描NFC标签"按钮
3. 将NFC标签靠近手机背面
4. 读取完成后，应用将显示标签中的数据内容

## 项目结构

```
├── App.uvue          # 应用根组件
├── main.js           # 应用入口文件
├── manifest.json     # 应用配置清单（包含NFC权限配置）
├── pages.json        # 页面路由配置
└── pages/
    └── index/        # 主页面
        └── index.uvue # 主页面组件
``` 