package main

import (
	"fmt"
	"github.com/otiai10/gosseract/v2"
	"gocv.io/x/gocv"
	"image"
)

func main() {
	src := gocv.IMRead("/home/ubuntu/personal-learning-record/golang/ocr/1746796663616.png", gocv.IMReadColor)
	if src.Empty() {
		fmt.Println("无法读取图片")
		return
	}
	defer src.Close()

	// 转为灰度图
	gray := gocv.NewMat()
	defer gray.Close()
	gocv.CvtColor(src, &gray, gocv.ColorBGRToGray)

	// 高斯模糊去噪
	blur := gocv.NewMat()
	defer blur.Close()
	gocv.GaussianBlur(gray, &blur, image.Pt(5, 5), 0, 0, gocv.BorderDefault)

	// 使用 Otsu 自动阈值进行二值化
	binary := gocv.NewMat()
	defer binary.Close()
	gocv.Threshold(blur, &binary, 0, 255, gocv.ThresholdBinary|gocv.ThresholdOtsu)

	// 可选：进行膨胀操作
	// kernel := gocv.GetStructuringElement(gocv.MorphRect, image.Pt(3, 3))
	// gocv.Dilate(binary, &binary, kernel)

	// 保存处理后的图像
	ok := gocv.IMWrite("processed.png", binary)
	if !ok {
		fmt.Println("保存处理图像失败")
		return
	}

	client := gosseract.NewClient()
	defer client.Close()
	//err := client.SetImageFromBytes()
	err := client.SetImage("/home/ubuntu/personal-learning-record/golang/ocr/processed.png")
	if err != nil {
		fmt.Println("err===11", err.Error())
		return
	}
	err = client.SetLanguage("eng")
	if err != nil {
		fmt.Println("err===22", err.Error())
		return
	}
	err = client.SetWhitelist("0123456789")
	if err != nil {
		fmt.Println("err===33", err.Error())
		return
	}
	text, err := client.Text()
	if err != nil {
		fmt.Println("err===", err.Error())
		return
	}
	fmt.Println("====", text)
}
