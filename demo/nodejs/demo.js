const CryptoJS = require("crypto-js");

// 固定密钥和 IV（必须 16 字节）
const AES_KEY = CryptoJS.enc.Utf8.parse("1234123412ABCDEF");  // 或替换成真实 key
const AES_IV = CryptoJS.enc.Utf8.parse("ABCDEF1234123412".substring(0, 16));


const qs = require("querystring");
const murmur = require("murmurhash3js");

const cryptoHelper = {
    /**
     * AES 解密（输入为 HEX 字符串）
     */
    decrypt: function (hexStr) {
        const hex = CryptoJS.enc.Hex.parse(hexStr);                     // HEX -> WordArray
        const base64Str = CryptoJS.enc.Base64.stringify(hex);           // WordArray -> Base64
        const decrypted = CryptoJS.AES.decrypt(base64Str, AES_KEY, {
            iv: AES_IV,
            mode: CryptoJS.mode.CBC,
            padding: CryptoJS.pad.Pkcs7
        });
        return decrypted.toString(CryptoJS.enc.Utf8);                   // 解密并转为字符串
    },

    /**
     * AES 加密（输出为大写 HEX 字符串）
     */
    encrypt: function (plainText) {
        const utf8Data = CryptoJS.enc.Utf8.parse(plainText);
        const encrypted = CryptoJS.AES.encrypt(utf8Data, AES_KEY, {
            iv: AES_IV,
            mode: CryptoJS.mode.CBC,
            padding: CryptoJS.pad.Pkcs7
        });
        return encrypted.ciphertext.toString().toUpperCase(); // HEX upper case
    },

    /**
     * 字符串转 Base64
     */
    toBase64: function (str) {
        return CryptoJS.enc.Base64.stringify(CryptoJS.enc.Utf8.parse(str));
    },

    /**
     * Base64 字符串转原始对象（UTF8 -> JSON.parse）
     */
    parseBase64: function (base64Str) {
        const decoded = CryptoJS.enc.Base64.parse(base64Str).toString(CryptoJS.enc.Utf8);
        return JSON.parse(decoded);
    }
};
console.log(cryptoHelper.decrypt("704C0B6213DBB9C4A5060FCE459E0639"))
module.exports = cryptoHelper;

/**
 * @param {Number} e - 时间戳
 * @param {Object} n - 请求配置对象（包含 url / data / params）
 * @param {Boolean} t - 是否为 GET 请求
 * @returns {String} - x-ca-nonce 的 128-bit 哈希签名
 */



// x-ca-nonce SDK
// 用于生成签名字段 x-ca-nonce，兼容 Node.js + CryptoJS + murmurhash3js


const XCaNonceSDK = {
    /**
     * AES 解密 hex 字符串，得到密钥（c）
     * @param {string} hexStr - 十六进制密文
     * @param {string} aesSecret - AES 密钥（如 "enduresurv1ve"）
     * @returns {string} 解密后的明文 key
     */
    decryptKey(hexStr, aesSecret) {
        const key = CryptoJS.enc.Utf8.parse(aesSecret);
        const iv = CryptoJS.enc.Utf8.parse(aesSecret.substring(0, 16));
        const hex = CryptoJS.enc.Hex.parse(hexStr);
        const base64 = CryptoJS.enc.Base64.stringify(hex);
        const decrypted = CryptoJS.AES.decrypt(base64, key, {
            iv: iv,
            mode: CryptoJS.mode.CBC,
            padding: CryptoJS.pad.Pkcs7
        });
        return decrypted.toString(CryptoJS.enc.Utf8);
    },

    /**
     * 生成 x-ca-nonce 哈希值
     * @param {object} options
     * @param {string} options.path - 请求路径
     * @param {object|null} options.params - GET 参数对象
     * @param {number} options.timestamp - 时间戳
     * @param {object|null} options.body - POST body
     * @param {string} options.encryptedKeyHex - 加密后的 key（十六进制）
     * @param {string} options.aesSecret - 解密用 AES 密钥
     * @param {boolean} options.isGet - 是否为 GET 请求
     * @returns {string} x-ca-nonce 签名值
     */
    generateNonce({ path, params, timestamp, body, encryptedKeyHex, isGet = true }) {
        const key = cryptoHelper.decrypt("704C0B6213DBB9C4A5060FCE459E0639");
        console.log(key)
        const fullPath = this.buildFullPath(path, params, isGet);
        console.log("fullPath",fullPath)
        // const bodyStr = !isGet && body ? JSON.stringify(this.stable(body)) : "";
        const bodyStr = !isGet && body ? JSON.stringify(body) : "";
        console.log("bodyStr====",bodyStr)
        const signStr = fullPath + "App" + timestamp + bodyStr + key;
        console.log("signStr====",signStr)
        const base64Str = cryptoHelper.toBase64(signStr);
        console.log("base64Str====",base64Str)
        console.log("murmur====",murmur)
        const result = murmur.x64.hash128(base64Str)
        console.log("result====",result)
        return result;
    },

    /**
     * 构建 URL，带 GET 参数或清理路径
     */
    buildFullPath(url, params, isGet) {
        if (isGet && params) {
            const query = serializeNestedQuery(params);
            return query ? `${url}?${query}` : url;
        }
        const cleaned = url.endsWith("?") ? url.slice(0, -1) : url;
        return encodeURI(cleaned);
    },

    /**
     * 将对象稳定排序后序列化
     */
    stable(obj) {
        return Object.keys(obj).sort().reduce((res, k) => {
            res[k] = obj[k];
            return res;
        }, {});
    },

    /**
     * GET 参数序列化函数
     */
    serializeQuery(params) {
        return Object.keys(params).sort().map(key => `${encodeURIComponent(key)}=${encodeURIComponent(params[key])}`).join("&");
    }
};
/**
 * 序列化对象为 URL 查询参数字符串（支持嵌套一层）
 *
 * @param {Object} obj - 参数对象
 * @returns {string} - query 字符串
 */
function serializeNestedQuery(obj) {
    const result = [];

    const keys = Object.keys(obj).sort();

    keys
        .filter(key => obj[key] !== undefined && (obj[key] || obj[key] === 0))
        .forEach(key => {
            const value = obj[key];

            if (Object.prototype.toString.call(value) === '[object Object]') {
                Object.keys(value)
                    .filter(subKey => value[subKey] || value[subKey] === 0)
                    .forEach(subKey => {
                        const compoundKey = `${key}[${subKey}]`;
                        result.push(`${encodeURIComponent(compoundKey)}=${encodeURIComponent(value[subKey])}`);
                    });
            } else {
                result.push(`${encodeURIComponent(key)}=${encodeURIComponent(value)}`);
            }
        });

    return result.join("&");
}


module.exports = XCaNonceSDK;
// const dee = XCaNonceSDK.generateNonce({
//   path: "/competition/luck/homePage/getHomeCompetitionLeagueMap",
//   // params: {lotteryName: '竞彩足球', shopId: 37947, userId: 100488596},
//   params: {lotteryName: '北京单场', shopId: 37947, userId: 100488596},
//   timestamp: 1746851319402,
//   body: {lotteryName: '竞彩足球', shopId: 37947, userId: 100488596},
//   encryptedKeyHex: "704C0B6213DBB9C4A5060FCE459E0639",
//   isGet: true
// });


const HeaderFieldHelper = {
    /**
     * 返回时间戳字段名
     */
    XCaTimestamp() {
        return "X-Ca-Timestamp";
    },

    /**
     * 返回平台字段名（可能用于标识来源端，如 app/web）
     */
    platform() {
        return "platform";
    },

    /**
     * 返回 nonce 字段名
     */
    XCaNonce() {
        return "X-Ca-Nonce";
    },

    /**
     * 返回 key 字段名（如 appid 或签名用 key）
     */
    XCaKey() {
        return "X-Ca-Key";
    }
};
module.exports = HeaderFieldHelper;

// function getConfig(config) {
//   // 添加 Authorization 头部
//   config.header["Authorization"] = uni.getStorageSync("access_token");

//   // 设置 x-ca-key（如 o = "91B713F1390C..."）
//   config.header[d.pName()] = o;

//   // 生成 x-ca-timestamp
//   const timestamp = Date.now();
//   config.header[d.tName()] = timestamp;

//   // 固定版本号和客户端类型
//   config.header["version"] = "1.1.8";
//   config.header["Client-Type"] = "user_h5";

//   // 是否启用安全加密（通过缓存判断）
//   const isSecurity = uni.getStorageSync("isSecurity") === "1";

//   // 请求方法
//   const method = config.method.toLowerCase();

//   if (method === "get") {
//     // 把 data 转为 params（GET 请求通常用 query 参数）
//     config.params = config.data;

//     // 生成 x-ca-nonce 签名（true 表示 GET 请求）
//     config.header[d.nName()] = r(timestamp, config, true);

//     // 若有参数，拼接到 URL 上，并清空 params
//     if (config.params) {
//       const queryString = s(config.params); // s() 为 query 编码函数（如 serializeNestedQuery）
//       config.url = `${config.url}?${queryString}`;
//       config.params = {};
//     }

//     // 如果开启加密，处理 GET 请求的加密逻辑
//     if (isSecurity) {
//       g(config, true); // g 是加密处理器
//     }

//   } else if (["post", "put", "delete"].includes(method)) {
//     // 生成 x-ca-nonce 签名（false 表示非 GET）
//     config.header[d.nName()] = r(timestamp, config, false);

//     // 如果开启加密，处理 POST 请求体加密逻辑
//     if (isSecurity) {
//       g(config, false);
//     }
//   }

//   return config;
// }

const axios = require("axios");
const { encrypt } = require("./xxtea");

const encryptedKeyHex = "704C0B6213DBB9C4A5060FCE459E0639"; // 加密密钥
const aesSecret = "enduresurv1ve"; // 用于解密 c
const accessToken = "eyJhbGciOiJIUzUxMiJ9.eyJ1c2VyX2lkIjoxMDA0ODg1OTYsInVzZXJfa2V5IjoiZDI4MTA5NjUtMDMxNi00OGM5LTg5NjktMWE0NjQ5OTFjYTExIiwiZGV2aWNlSWQiOiIxNzQ2Nzg3OTIwNDUyNjU5MDkzIiwicGxhdGZvcm0iOiJBcHAiLCJ1c2VybmFtZSI6IjEzNjU3NzI2ODU2In0.RpMHgGpiCmJp5UFVjqB8BYTL7Jye70bxFSkU3A6wRsaY9Moo8bWDD8t2lkKLONGHSt0ApvrG-jr3WP3h2szYoA";

async function sendRequest() {
    const timestamp = Date.now();
    // const timestamp = 1746869770961;
    // const path = "/system/shop/shop";
    // const path = "/competition/luck/homePage/userInfo";
    const path = "/competition/luck/homePage/getHomeCompetitionLeagueMap";
    // const params = {lotteryName:"竞彩足球",shopId:37947,userId:100488596 };
    // const params = {lotteryName:"北京单场",shopId:37947,userId:100488596 };
    const params = {lotteryName:"北京单场",shopId:37947,userId:100488596 };
    // const params = {  shopId:37947,userId:100488596};
    const isGet = true;
//重新登录
    // 生成 x-ca-nonce 签名
    const xCaNonce = XCaNonceSDK.generateNonce({
        path,
        params,
        timestamp,
        body: {"mobile":"13657726856","password":"aa12345678","code":"0375","uuid":"ea25864eefff45b5ad151f59cca11344","version":"2.0.0","deviceId":"17468408185848999827","equipmentModel":"android-","deviceSource":"android"},
        encryptedKeyHex,
        aesSecret,
        isGet
    });

    const https = require("https");

    const agent = new https.Agent({ rejectUnauthorized: false });


    const headers = {
        Authorization: accessToken,
        [HeaderFieldHelper.XCaKey()]: Math.random().toString(16).slice(3),
        [HeaderFieldHelper.XCaTimestamp()]: timestamp,
        [HeaderFieldHelper.XCaNonce()]: xCaNonce,
        [HeaderFieldHelper.platform()]: "App",
        "host":"tengxnewauser003.hhjkkz.com",
        "referer":"https://tengxnewauser003.hhjkkz.com/",
        "user-agent":"Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Mobile Safari/537.36",
        "Client-Type": "user_h5",
        version: "1.1.8"
    };

    const queryString = Object.keys(params)
        .sort()
        .map(key => `${encodeURIComponent(key)}=${encodeURIComponent(params[key])}`)
        .join("&");

    console.log("queryString",queryString);
    const paramsResult=encrypt(queryString,aesSecret)
    const paramsResultURI=encodeURIComponent(paramsResult)
    const baseUrl = "https://tengxnewauser003.hhjkkz.com/prod-api"; // 修改为真实 API 域名
    const url = baseUrl + path+"?"+paramsResultURI;

    console.log("headers====",headers)
    console.log("url====",url)
    // 获取 游戏列表
    // https://tengxnewauser003.hhjkkz.com/prod-api/competition/luck/homePage/getGameList?SKCn4QeTsq%2F4xrzu28AC7w%3D%3D
    try {
        const res = await axios.get(url, { headers, httpsAgent: agent  });
        console.log("✅ 请求成功：",JSON.stringify(res.data,null, 2));
    } catch (err) {
        console.error("❌ 请求失败：", err.response?.data || err.message);
    }
}

// sendRequest();
//=========================================
async function postRequest() {

    const timestamp = Date.now();
    // const timestamp = 1746871108534;
    // const path = "/system/shop/shop";
    // const path = "/competition/luck/homePage/userInfo";
    const path = "/auth/app/newLogin";
    // const params = {lotteryName:"竞彩足球",shopId:37947,userId:100488596 };
    // const params = {lotteryName:"北京单场",shopId:37947,userId:100488596 };
    // const params = {lotteryName:"北京单场",shopId:37947,userId:100488596 };
    const params = { };
    // const params = {  shopId:37947,userId:100488596};
    const isGet = false;
//重新登录
    let data = {
        "mobile": "13657726856",
        "password": "aa12345678",
        "code": "8795",
        "uuid": "b155e7f964aa46b4b505782a9468a93e",
        "version": "2.0.0",
        "deviceId": "17468408185848999827",
        "equipmentModel": "android-",
        "deviceSource": "android"
    }
    // 生成 x-ca-nonce 签名
    const xCaNonce = XCaNonceSDK.generateNonce({
        path,
        params,
        timestamp,
        body: data,
        encryptedKeyHex,
        aesSecret,
        isGet
    });

    const https = require("https");

    const agent = new https.Agent({ rejectUnauthorized: false });


    const headers = {
        Authorization: accessToken,
        [HeaderFieldHelper.XCaKey()]: Math.random().toString(16).slice(3),
        [HeaderFieldHelper.XCaTimestamp()]: timestamp,
        [HeaderFieldHelper.XCaNonce()]: xCaNonce,
        [HeaderFieldHelper.platform()]: "App",
        "host":"tengxnewauser003.hhjkkz.com",
        "referer":"https://tengxnewauser003.hhjkkz.com/",
        "user-agent":"Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Mobile Safari/537.36",
        "Client-Type": "user_h5",
        'Content-Type': 'application/json',
        version: "1.1.8"
    };

    const dataJson = JSON.stringify(data)
    const bodyReq=encrypt(dataJson,aesSecret)

    const baseUrl = "https://tengxnewauser003.hhjkkz.com/prod-api"; // 修改为真实 API 域名
    const url = baseUrl + path;

    console.log("headers====",headers)
    console.log("url====",url)
    console.log("bodyReq====",bodyReq)

    try {
        const response = await axios.post(url, bodyReq, { headers ,httpsAgent: agent });
        console.log('✅ 请求成功POST:', response.data);
    } catch (error) {
        console.error('❌ 请求失败POST:', error.response?.data || error.message);
    }
}
postRequest()