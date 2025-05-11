package main

/*
#cgo CFLAGS: `pkg-config --cflags glib-2.0 gio-2.0`
#cgo LDFLAGS: `pkg-config --libs glib-2.0 gio-2.0`
#include <glib.h>
#include <gio/gio.h>
*/
import (
	"C"
	"fmt"
	"github.com/otiai10/gosseract/v2"
	"gocv.io/x/gocv"
)

//func main() {
//	src := gocv.IMRead("/home/ubuntu/personal-learning-record/golang/ocr/1746796663616.png", gocv.IMReadColor)
//	if src.Empty() {
//		fmt.Println("无法读取图片")
//		return
//	}
//	defer src.Close()
//
//	//// 转为灰度图
//	//gray := gocv.NewMat()
//	//defer gray.Close()
//	//gocv.CvtColor(src, &gray, gocv.ColorGrayToRGB)
//	//
//	//// 高斯模糊去噪
//	//blur := gocv.NewMat()
//	//defer blur.Close()
//	//gocv.GaussianBlur(gray, &blur, image.Pt(5, 5), 0, 0, gocv.BorderDefault)
//	//
//	//// 使用 Otsu 自动阈值进行二值化
//	//binary := gocv.NewMat()
//	//defer binary.Close()
//	//gocv.Threshold(blur, &binary, 0, 255, gocv.ThresholdBinary|gocv.ThresholdOtsu)
//	//
//	//// 可选：进行膨胀操作
//	//// kernel := gocv.GetStructuringElement(gocv.MorphRect, image.Pt(3, 3))
//	//// gocv.Dilate(binary, &binary, kernel)
//	//
//	//// 保存处理后的图像
//	//ok := gocv.IMWrite("processed.png", binary)
//	//if !ok {
//	//	fmt.Println("保存处理图像失败")
//	//	return
//	//}
//
//	client := gosseract.NewClient()
//	defer client.Close()
//	//err := client.SetImageFromBytes()
//	//err := client.SetImage("/home/ubuntu/personal-learning-record/golang/ocr/processed.png")
//	//err := client.SetImage("/home/ubuntu/personal-learning-record/golang/ocr/1746796663616.png")
//	//if err != nil {
//	//	fmt.Println("err===11", err.Error())
//	//	return
//	//}
//	err := client.SetLanguage("eng")
//	if err != nil {
//		fmt.Println("err===22", err.Error())
//		return
//	}
//	err = client.SetWhitelist("0123456789")
//	if err != nil {
//		fmt.Println("err===33", err.Error())
//		return
//	}
//	// 将图片转换为灰度图
//	gray := gocv.NewMat()
//	gocv.CvtColor(src, &gray, gocv.ColorBGRToGray)
//	defer gray.Close()
//
//	// 使用 Tesseract 进行 OCR
//	text, err := client.SetImageFromBytes(gray.ToImage())
//	if err != nil {
//		log.Fatalf("OCR 错误: %v", err)
//	}
//
//	fmt.Printf("识别到的数字是: %s\n", text)
//
//	//text, err := client.Text()
//	//if err != nil {
//	//	fmt.Println("err===", err.Error())
//	//	return
//	//}
//	//fmt.Println("====", text)
//}

func main() {
	// 加载图片
	img := gocv.IMRead("/home/ubuntu/personal-learning-record/golang/ocr/1746796663616.png", gocv.IMReadColor)
	if img.Empty() {
		fmt.Println("无法加载图片")
		return
	}
	defer img.Close()

	// 将图片转换为灰度图
	gray := gocv.NewMat()
	gocv.CvtColor(img, &gray, gocv.ColorBGRToGray)
	defer gray.Close()

	// 使用 Gosseract 创建客户端
	client := gosseract.NewClient()
	defer client.Close()

	// 设置语言和白名单（仅允许数字）
	client.SetLanguage("eng")
	client.SetWhitelist("0123456789")

	// 写入临时文件以供 Gosseract 处理
	tmpFile := "temp_image.png"
	gocv.IMWrite(tmpFile, gray)

	// 执行 OCR 并获取结果
	err := client.SetImage(tmpFile)
	if err != nil {
		fmt.Printf("OCR 错误: %v\n", err)
		return
	}
	text, err := client.Text()
	if err != nil {
		fmt.Println("err===", err.Error())
		return
	}
	fmt.Println("====", text)
	fmt.Printf("识别到的数字是: %s\n", text)
}
