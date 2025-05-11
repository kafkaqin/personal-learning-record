package main

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"fmt"
	"log"
)

func toBytes(str string) []byte {
	return []byte(str)
}

func toString(data []byte) string {
	return string(data)
}

func toUint32Array(data []byte, includeLength bool) []uint32 {
	n := (len(data) + 3) / 4
	var result []uint32
	if includeLength {
		result = make([]uint32, n+1)
		result[n] = uint32(len(data))
	} else {
		result = make([]uint32, n)
	}
	for i := 0; i < len(data); i++ {
		result[i/4] |= uint32(data[i]) << ((i % 4) * 8)
	}
	return result
}

func fromUint32Array(data []uint32, includeLength bool) []byte {
	var length int
	if includeLength {
		length = int(data[len(data)-1])
		data = data[:len(data)-1]
	} else {
		length = len(data) * 4
	}
	buffer := new(bytes.Buffer)
	for _, val := range data {
		_ = binary.Write(buffer, binary.LittleEndian, val)
	}
	return buffer.Bytes()[:length]
}

func encryptUint32Array(v []uint32, k []uint32) []uint32 {
	n := len(v)
	if n < 2 {
		return v
	}
	var (
		z     = v[n-1]
		y     uint32
		sum   uint32
		delta = uint32(0x9e3779b9)
		q     = 6 + 52/n
	)
	for i := 0; i < q; i++ {
		sum += delta
		e := (sum >> 2) & 3
		for p := 0; p < n; p++ {
			y = v[(p+1)%n]
			mx := (((z >> 5) ^ (y << 2)) + ((y >> 3) ^ (z << 4))) ^ ((sum ^ y) + (k[(uint32(p)&3)^e] ^ z))
			v[p] += mx
			z = v[p]
		}
	}
	return v
}

func decryptUint32Array(v []uint32, k []uint32) []uint32 {
	n := len(v)
	if n < 2 {
		return v
	}
	var (
		y     = v[0]
		z     uint32
		delta = uint32(0x9e3779b9)
		q     = 6 + 52/n
		sum   = delta * uint32(q)
	)
	for sum != 0 {
		e := (sum >> 2) & 3
		for p := n - 1; p >= 0; p-- {
			z = v[(p-1+n)%n]
			mx := (((z >> 5) ^ (y << 2)) + ((y >> 3) ^ (z << 4))) ^ ((sum ^ y) + (k[(uint32(p)&3)^e] ^ z))
			v[p] -= mx
			y = v[p]
		}
		sum -= delta
	}
	return v
}

func Encrypt(data, key string) string {
	v := toUint32Array(toBytes(data), true)
	k := toUint32Array(toBytes(key), false)
	if len(k) < 4 {
		// pad key to 4 uint32s
		newK := make([]uint32, 4)
		copy(newK, k)
		k = newK
	}
	encrypted := encryptUint32Array(v, k)
	return base64.StdEncoding.EncodeToString(fromUint32Array(encrypted, false))
}

func Decrypt(base64Str, key string) (string, error) {
	decoded, err := base64.StdEncoding.DecodeString(base64Str)
	if err != nil {
		return "", err
	}
	v := toUint32Array(decoded, false)
	k := toUint32Array(toBytes(key), false)
	if len(k) < 4 {
		newK := make([]uint32, 4)
		copy(newK, k)
		k = newK
	}
	decrypted := decryptUint32Array(v, k)
	bytes := fromUint32Array(decrypted, true)
	return toString(bytes), nil
}
func main() {

	key := "enduresurv1ve"
	data := `{"mobile": "13657726856","password": "aa12345678","code": "8795","uuid": "b155e7f964aa46b4b505782a9468a93e","version": "2.0.0","deviceId": "17468408185848999827","equipmentModel": "android-","deviceSource": "android"}`

	encrypted := Encrypt(data, key)
	fmt.Println("Encrypted:", encrypted)

	decrypted, err := Decrypt(encrypted, key)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("Decrypted:", decrypted)
}
