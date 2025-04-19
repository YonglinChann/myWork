const __uniConfig = {"pages":[],"globalStyle":{"navigationBarTextStyle":"white","navigationBarTitleText":"NFC标签读取应用","navigationBarBackgroundColor":"#007aff","backgroundColor":"#F8F8F8"},"appname":"NFC读取应用","compilerVersion":"4.57","entryPagePath":"pages/index/index","entryPageQuery":"","realEntryPagePath":"","themeConfig":{}};
__uniConfig.getTabBarConfig = () =>  {return undefined};
__uniConfig.tabBar = __uniConfig.getTabBarConfig();
const __uniRoutes = [{"path":"pages/index/index","meta":{"isQuit":true,"isEntry":true,"navigationBarTitleText":"NFC标签读取应用","navigationBarBackgroundColor":"#007aff","navigationBarTextStyle":"white"}}].map(uniRoute=>(uniRoute.meta.route=uniRoute.path,__uniConfig.pages.push(uniRoute.path),uniRoute.path='/'+uniRoute.path,uniRoute)).concat(typeof __uniSystemRoutes !== 'undefined' ? __uniSystemRoutes : []);

