
// 图片路径
const imageFilePath = '/home/ubuntu/personal-learning-record/golang/ocr/1746796663616.png';
const Tesseract = require('tesseract.js');
const sharp = require('sharp');

// 图片路径

// 使用 sharp 进行图像预处理
sharp(imageFilePath)
    .greyscale() // 转换为灰度图
    .threshold(128) // 二值化
    .morphology('erode', 'square', 3) // 腐蚀操作，减少噪声
    .morphology('dilate', 'square', 3) // 膨胀操作，恢复字符形状
    .toFile('processed_image.png', (err) => {
        if (err) {
            console.error('图像预处理出错:', err);
            return;
        }

        // 执行 OCR
        Tesseract.recognize(
            'processed_image.png',
            'eng', // 设置为英文模式
            {
                tessedit_char_whitelist: '0123456789', // 只允许数字
                psm: 6, // Page Segmentation Mode: 假设图像是一个单行文本
            }
        )
            .then(({ data: { text } }) => {
                console.log(`识别到的数字是: ${text}`);
            })
            .catch(err => {
                console.error('OCR 处理出错:', err);
            });
    });