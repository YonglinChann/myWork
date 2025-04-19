//
//  DCUniAdMediatedNativeAd.h
//  DCUniAdFoundation
//
//  Created by wangzhitong on 2024/7/30.
//

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>
NS_ASSUME_NONNULL_BEGIN

@interface DCUniAdMediatedNativeAd : NSObject

// 广告视图对象，一般指原生广告的根视图
@property (nonatomic, strong) UIView *view;

// adn获取到的原始广告数据 [Required] (adn广告回调的原始数据，通过该数据建立对应关系)
@property (nonatomic, strong) id originMediatedNativeAd;

@end

NS_ASSUME_NONNULL_END
