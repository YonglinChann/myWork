/**
 * NFT之钥 应用下载页设备检测脚本
 * 用于识别不同品牌的安卓手机并提供相应的下载链接
 */

document.addEventListener('DOMContentLoaded', function() {
    // 获取页面元素
    const deviceBrandEl = document.getElementById('device-brand');
    const deviceModelEl = document.getElementById('device-model');
    const downloadBtn = document.getElementById('download-btn');
    const downloadDesc = document.getElementById('download-description');
    const recommendedDownload = document.getElementById('recommended-download');
    const otherDownloads = document.querySelector('.other-downloads');
    
    // 获取 User-Agent 字符串
    const userAgent = navigator.userAgent;
    console.log('用户设备 User-Agent:', userAgent);
    
    // 品牌信息配置
    const brandInfo = {
        huawei: {
            name: '华为',
            downloadUrl: 'https://appgallery.huawei.com/app/NFT之钥',
            color: '#cf0a2c',
            description: '通过华为应用市场下载，获得最佳体验'
        },
        xiaomi: {
            name: '小米',
            downloadUrl: 'https://app.mi.com/NFT之钥',
            color: '#ff6700',
            description: '通过小米应用商店下载，获得最佳体验'
        },
        oppo: {
            name: 'OPPO',
            downloadUrl: 'https://store.oppo.com/app/NFT之钥',
            color: '#008000',
            description: '通过 OPPO 软件商店下载，获得最佳体验'
        },
        vivo: {
            name: 'vivo',
            downloadUrl: 'https://apps.vivo.com.cn/NFT之钥',
            color: '#415fff',
            description: '通过 vivo 应用商店下载，获得最佳体验'
        },
        iqoo: {
            name: 'iQOO',
            downloadUrl: 'https://apps.vivo.com.cn/NFT之钥',
            color: '#0078FF',
            description: 'iQOO设备请通过 vivo 应用商店下载，获得最佳体验'
        },
        samsung: {
            name: '三星',
            downloadUrl: 'https://www.example.com/download/android/NFT之钥.apk',
            color: '#1428a0',
            description: '适用于三星设备的通用安卓版本'
        },
        oneplus: {
            name: 'OnePlus',
            downloadUrl: 'https://store.oppo.com/app/NFT之钥',
            color: '#f5010c',
            description: '一加设备请通过 OPPO 软件商店下载，获得最佳体验'
        },
        meizu: {
            name: '魅族',
            downloadUrl: 'https://www.example.com/download/android/NFT之钥.apk',
            color: '#008cff',
            description: '适用于魅族设备的通用安卓版本'
        },
        generic: {
            name: '安卓',
            downloadUrl: 'https://www.example.com/download/android/NFT之钥.apk',
            color: '#3ddc84',
            description: '通用安卓版本'
        },
        pc: {
            name: '电脑',
            downloadUrl: '#',
            color: '#444444',
            description: '请使用手机设备访问获得最佳体验'
        }
    };
    
    // 检测设备品牌和类型
    function detectDevice() {
        let detectedBrand = 'generic'; // 默认为通用安卓
        let isPC = false;
        
        // 检测是否为PC设备 (Windows, Mac, Linux)
        if (/Windows NT|Macintosh|Linux(?!.*Android)/i.test(userAgent) && !/Mobile|Android|iPhone|iPad|iPod/i.test(userAgent)) {
            isPC = true;
            detectedBrand = 'pc';
        }
        // 识别是否为iOS设备
        else if (/iPhone|iPad|iPod/i.test(userAgent)) {
            deviceBrandEl.textContent = '检测到 iOS 设备';
            deviceModelEl.style.display = 'none'; // 隐藏设备型号元素
            downloadBtn.href = 'https://apps.apple.com/app/NFT之钥';
            downloadBtn.textContent = '前往 App Store';
            downloadDesc.textContent = '通过 App Store 下载 iOS 版本';
            return;
        }
        // 检测安卓品牌
        else {
            const brandPatterns = {
                huawei: /HUAWEI|HONOR/i,
                xiaomi: /Mi|Redmi|Xiaomi|POCO/i,
                oppo: /OPPO|PAFM|PAHM|PBBM|PBCM/i,
                vivo: /vivo(?!\/iQOO)/i,  // vivo但非iQOO
                iqoo: /iQOO|vivo\/iQOO/i, // 识别iQOO品牌
                samsung: /Samsung/i,
                oneplus: /OnePlus/i,
                meizu: /Meizu|M(?:\d{1,3})Note/i,
            };
            
            // 遍历检测各品牌
            for (const [brand, pattern] of Object.entries(brandPatterns)) {
                if (pattern.test(userAgent)) {
                    detectedBrand = brand;
                    break;
                }
            }
        }
        
        // 更新 UI
        if (brandInfo[detectedBrand]) {
            const brand = brandInfo[detectedBrand];
            deviceBrandEl.textContent = `识别到您正在使用「${brand.name}」设备`;
            deviceBrandEl.style.color = brand.color;
            
            // 隐藏设备型号信息
            deviceModelEl.style.display = 'none';
            
            // 如果是PC，改变显示方式
            if (isPC) {
                // 隐藏推荐下载区域或调整其样式
                recommendedDownload.style.display = 'none';
                
                // 调整其他下载选项区域的样式
                otherDownloads.style.marginTop = '30px';
                const otherTitle = otherDownloads.querySelector('h3');
                if (otherTitle) {
                    otherTitle.textContent = '请选择下载方式';
                    otherTitle.style.fontSize = '1.5rem';
                    otherTitle.style.fontWeight = 'bold';
                    otherTitle.style.marginBottom = '20px';
                }
                
                // 可以添加PC访问提示
                const pcMessage = document.createElement('p');
                pcMessage.textContent = '我们检测到您正在使用电脑访问，请使用手机扫描下方二维码或选择合适的下载方式';
                pcMessage.style.marginBottom = '20px';
                pcMessage.style.color = '#666';
                otherDownloads.insertBefore(pcMessage, otherDownloads.querySelector('.download-links'));
                
                // 如果需要的话，这里可以添加二维码图片
                // const qrCode = document.createElement('img');
                // qrCode.src = 'images/download-qr.png';
                // qrCode.alt = '下载二维码';
                // qrCode.style.width = '200px';
                // qrCode.style.height = '200px';
                // qrCode.style.margin = '0 auto 20px';
                // qrCode.style.display = 'block';
                // otherDownloads.insertBefore(qrCode, otherDownloads.querySelector('.download-links'));
            } else {
                downloadBtn.href = brand.downloadUrl;
                downloadDesc.textContent = brand.description;
                
                // 如果是一加设备，高亮显示OPPO下载链接
                if (detectedBrand === 'oneplus') {
                    const brandLinks = document.querySelectorAll('.brand-link');
                    brandLinks.forEach(link => {
                        if (link.getAttribute('data-brand') === 'oppo') {
                            link.classList.add('oppo-color');
                            link.style.fontWeight = 'bold';
                        }
                    });
                }
                
                // 如果是iQOO设备，高亮显示vivo下载链接
                if (detectedBrand === 'iqoo') {
                    const brandLinks = document.querySelectorAll('.brand-link');
                    brandLinks.forEach(link => {
                        if (link.getAttribute('data-brand') === 'vivo') {
                            link.classList.add('vivo-color');
                            link.style.fontWeight = 'bold';
                        }
                    });
                }
            }
            
            // 高亮对应品牌的下载链接
            const brandLinks = document.querySelectorAll('.brand-link');
            brandLinks.forEach(link => {
                if (link.getAttribute('data-brand') === detectedBrand && !isPC && 
                    detectedBrand !== 'oneplus' && detectedBrand !== 'iqoo') {
                    link.classList.add(detectedBrand + '-color');
                    link.style.fontWeight = 'bold';
                }
                
                // 设置每个链接的正确跳转地址
                const linkBrand = link.getAttribute('data-brand');
                if (brandInfo[linkBrand]) {
                    link.href = brandInfo[linkBrand].downloadUrl;
                }
            });
        }
    }
    
    // 执行设备检测
    detectDevice();
    
    // 为所有品牌链接设置正确的链接
    document.querySelectorAll('.brand-link').forEach(link => {
        const brand = link.getAttribute('data-brand');
        if (brandInfo[brand]) {
            link.href = brandInfo[brand].downloadUrl;
        }
    });
}); 