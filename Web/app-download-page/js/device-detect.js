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
        samsung: {
            name: '三星',
            downloadUrl: 'https://www.example.com/download/android/NFT之钥.apk',
            color: '#1428a0',
            description: '适用于三星设备的通用安卓版本'
        },
        oneplus: {
            name: 'OnePlus',
            downloadUrl: 'https://www.example.com/download/android/NFT之钥.apk',
            color: '#f5010c',
            description: '适用于OnePlus设备的通用安卓版本'
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
        }
    };
    
    // 检测设备品牌和型号
    function detectDevice() {
        let detectedBrand = 'generic'; // 默认为通用安卓
        let modelInfo = '';
        
        // 识别是否为iOS设备
        if (/iPhone|iPad|iPod/i.test(userAgent)) {
            deviceBrandEl.textContent = '检测到 iOS 设备';
            deviceModelEl.textContent = '请前往 App Store 下载';
            downloadBtn.href = 'https://apps.apple.com/app/NFT之钥';
            downloadBtn.textContent = '前往 App Store';
            downloadDesc.textContent = '通过 App Store 下载 iOS 版本';
            return;
        }
        
        // 检测安卓品牌
        const brandPatterns = {
            huawei: /HUAWEI|HONOR/i,
            xiaomi: /Mi|Redmi|Xiaomi|POCO/i,
            oppo: /OPPO|PAFM|PAHM|PBBM|PBCM/i,
            vivo: /vivo/i,
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
        
        // 尝试提取手机型号信息（简单版本）
        const modelMatches = userAgent.match(/Android.*?;\s*(?:[\w-]+\s*)+?(?=Build\/|;|\))/i);
        if (modelMatches && modelMatches[0]) {
            // 简单清理获取到的型号信息
            modelInfo = modelMatches[0].replace(/Android.*?;\s*/i, '').trim();
        }
        
        // 更新 UI
        if (brandInfo[detectedBrand]) {
            const brand = brandInfo[detectedBrand];
            deviceBrandEl.textContent = `识别到您的手机可能为「${brand.name}」设备`;
            deviceBrandEl.style.color = brand.color;
            
            if (modelInfo) {
                deviceModelEl.textContent = modelInfo;
            }
            
            downloadBtn.href = brand.downloadUrl;
            downloadDesc.textContent = brand.description;
            
            // 高亮对应品牌的下载链接
            const brandLinks = document.querySelectorAll('.brand-link');
            brandLinks.forEach(link => {
                if (link.getAttribute('data-brand') === detectedBrand) {
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
    
    // 如果浏览器支持更详细的设备信息 API
    function getMoreDetailedInfo() {
        // 如果浏览器支持 navigator.userAgentData
        if (navigator.userAgentData && navigator.userAgentData.getHighEntropyValues) {
            navigator.userAgentData.getHighEntropyValues(['platform', 'platformVersion', 'model'])
                .then(data => {
                    console.log('高熵值信息:', data);
                    if (data.model) {
                        deviceModelEl.textContent = data.model;
                    }
                })
                .catch(error => console.log('无法获取高熵值信息:', error));
        }
    }
    
    // 执行设备检测
    detectDevice();
    
    // 尝试获取更详细的信息（如果浏览器支持）
    getMoreDetailedInfo();
    
    // 为所有品牌链接设置正确的链接
    document.querySelectorAll('.brand-link').forEach(link => {
        const brand = link.getAttribute('data-brand');
        if (brandInfo[brand]) {
            link.href = brandInfo[brand].downloadUrl;
        }
    });
}); 