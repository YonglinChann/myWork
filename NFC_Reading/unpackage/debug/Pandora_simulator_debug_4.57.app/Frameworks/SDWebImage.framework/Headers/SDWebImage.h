/*
 * This file is part of the SDWebImage package.
 * (c) Olivier Poitrey <rs@dailymotion.com>
 * (c) Florent Vilmart
 *
 * For the full copyright and license information, please view the LICENSE
 * file that was distributed with this source code.
 */

#import <SDWebImage/SDWebImageCompat.h>

//! Project version number for SDWebImage.
FOUNDATION_EXPORT double SDWebImageVersionNumber;

//! Project version string for SDWebImage.
FOUNDATION_EXPORT const unsigned char SDWebImageVersionString[];

// In this header, you should import all the public headers of your framework using statements like #import <SDWebImage/PublicHeader.h>

#import <SDWebImage/SDWebImageManager.h>
#import <SDWebImage/SDCallbackQueue.h>
#import <SDWebImage/SDWebImageCacheKeyFilter.h>
#import <SDWebImage/SDWebImageCacheSerializer.h>
#import <SDWebImage/SDImageCacheConfig.h>
#import <SDWebImage/SDImageCache.h>
#import <SDWebImage/SDMemoryCache.h>
#import <SDWebImage/SDDiskCache.h>
#import <SDWebImage/SDImageCacheDefine.h>
#import <SDWebImage/SDImageCachesManager.h>
#import <SDWebImage/UIView+WebCache.h>
#import <SDWebImage/UIImageView+WebCache.h>
#import <SDWebImage/UIImageView+HighlightedWebCache.h>
#import <SDWebImage/SDWebImageDownloaderConfig.h>
#import <SDWebImage/SDWebImageDownloaderOperation.h>
#import <SDWebImage/SDWebImageDownloaderRequestModifier.h>
#import <SDWebImage/SDWebImageDownloaderResponseModifier.h>
#import <SDWebImage/SDWebImageDownloaderDecryptor.h>
#import <SDWebImage/SDImageLoader.h>
#import <SDWebImage/SDImageLoadersManager.h>
#import <SDWebImage/UIButton+WebCache.h>
#import <SDWebImage/SDWebImagePrefetcher.h>
#import <SDWebImage/UIView+WebCacheOperation.h>
#import <SDWebImage/UIView+WebCacheState.h>
#import <SDWebImage/UIImage+Metadata.h>
#import <SDWebImage/UIImage+MultiFormat.h>
#import <SDWebImage/UIImage+MemoryCacheCost.h>
#import <SDWebImage/UIImage+ExtendedCacheData.h>
#import <SDWebImage/SDWebImageOperation.h>
#import <SDWebImage/SDWebImageDownloader.h>
#import <SDWebImage/SDWebImageTransition.h>
#import <SDWebImage/SDWebImageIndicator.h>
#import <SDWebImage/SDImageTransformer.h>
#import <SDWebImage/UIImage+Transform.h>
#import <SDWebImage/SDAnimatedImage.h>
#import <SDWebImage/SDAnimatedImageView.h>
#import <SDWebImage/SDAnimatedImageView+WebCache.h>
#import <SDWebImage/SDAnimatedImagePlayer.h>
#import <SDWebImage/SDImageCodersManager.h>
#import <SDWebImage/SDImageCoder.h>
#import <SDWebImage/SDImageAPNGCoder.h>
#import <SDWebImage/SDImageGIFCoder.h>
#import <SDWebImage/SDImageIOCoder.h>
#import <SDWebImage/SDImageFrame.h>
#import <SDWebImage/SDImageCoderHelper.h>
#import <SDWebImage/SDImageGraphics.h>
#import <SDWebImage/SDGraphicsImageRenderer.h>
#import <SDWebImage/UIImage+GIF.h>
#import <SDWebImage/UIImage+ForceDecode.h>
#import <SDWebImage/NSData+ImageContentType.h>
#import <SDWebImage/SDWebImageDefine.h>
#import <SDWebImage/SDWebImageError.h>
#import <SDWebImage/SDWebImageOptionsProcessor.h>
#import <SDWebImage/SDImageIOAnimatedCoder.h>
#import <SDWebImage/SDImageHEICCoder.h>
#import <SDWebImage/SDImageAWebPCoder.h>
#import <SDWebImage/SDImageWebPCoder.h>

// Mac
#if __has_include(<SDWebImage/NSImage+Compatibility.h>)
#import <SDWebImage/NSImage+Compatibility.h>
#endif
#if __has_include(<SDWebImage/NSButton+WebCache.h>)
#import <SDWebImage/NSButton+WebCache.h>
#endif
#if __has_include(<SDWebImage/SDAnimatedImageRep.h>)
#import <SDWebImage/SDAnimatedImageRep.h>
#endif

// MapKit
#if __has_include(<SDWebImage/MKAnnotationView+WebCache.h>)
#import <SDWebImage/MKAnnotationView+WebCache.h>
#endif
