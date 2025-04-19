//
//  DCUniAdNativeAd.h
//  DCUniAdFoundation
//
//  Created by wangzhitong on 2024/7/31.
//

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>
#import "DCUniAdEnum.h"

@class DCUniAdNativeAd;
@class DCUniAdDislikeWords;
NS_ASSUME_NONNULL_BEGIN

@protocol DCUniAdNativeAdDelegate <NSObject>
@optional
/**
 * 原生模板广告渲染成功, 此时的 nativeExpressAdView.size.height 根据 size.width 完成了动态更新。
 */
- (void)nativeAdExpressViewRenderSuccess:(DCUniAdNativeAd *)nativeAd;

/**
 * 原生模板广告渲染失败
 */
- (void)nativeAdExpressViewRenderFail:(DCUniAdNativeAd *)nativeAd error:(NSError *)error;

/**
 广告曝光回调

 @param nativeAd DCUniAdNativeAd 实例
 */
- (void)nativeAdWillExpose:(DCUniAdNativeAd *)nativeAd;

/**
 广告点击回调

 @param nativeAd DCUniAdNativeAd 实例
 */
- (void)nativeAdDidClick:(DCUniAdNativeAd *)nativeAd;

/**
 广告详情页关闭回调

 @param nativeAd DCUniAdNativeAd 实例
 */
- (void)nativeAdDetailViewClosed:(DCUniAdNativeAd *)nativeAd;

/**
 广告详情页面即将展示回调

 @param nativeAd DCUniAdNativeAd 实例
 */
- (void)nativeAdDetailViewWillPresentScreen:(DCUniAdNativeAd *)nativeAd;

/**
 视频广告播放状态更改回调

 @param nativeAd DCUniAdNativeAd 实例
 @param status 视频广告播放状态
 */
- (void)nativeAd:(DCUniAdNativeAd *)nativeAd playerStatusChanged:(DCUniAdMediaPlayerStatus)status;


/**
 点击dislike回调
 开发者需要在这个回调中移除视图，否则，会出现用户点击叉无效的情况
 
 @param filterWords : 选择不喜欢的原因
 */
- (void)nativeAd:(DCUniAdNativeAd *)nativeAd dislikeWithReason:(NSArray<DCUniAdDislikeWords *> *)filterWords;

@end

@interface DCUniAdNativeAd : NSObject

/// 广告类型（Feed/Draw）
@property (nonatomic, assign, readonly) DCUniAdNativeAdSlotAdType adType;

/// 广告代理对象
@property (nonatomic, weak) id<DCUniAdNativeAdDelegate> delegate;

/// 广告点击弹出新页面所依赖的ViewController
@property (nonatomic, weak) UIViewController *viewController;

/// 模版视图
@property (nonatomic, weak, readonly) UIView *expressView;

/// 模版广告开始渲染
- (void)render;


/// drawAd注册广告视图
/// - Parameter containerView: 原生广告的容器视图
- (void)registerContainer:(UIView *)containerView;


/// drawAd取消注册视图
- (void)unregisterView;


@end

NS_ASSUME_NONNULL_END
