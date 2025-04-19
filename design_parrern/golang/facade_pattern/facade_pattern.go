package main

import "fmt"

/**
外观设计模式（Facade Pattern）是一种结构型设计模式，它为复杂子系统提供一个统一的接口，
使得子系统更易使用，隐藏内部细节、降低耦合度。
外观模式就像咖啡机的按钮，屏蔽复杂内部逻辑，提供一个简单的入口。
 模式作用
简化客户端调用，对外提供一个一致、简洁的接口。
屏蔽子系统细节，客户端无需了解内部子模块。
降低耦合性，模块之间解耦，便于维护和扩展。
*/

type DVDPlayer struct {
}

func (p *DVDPlayer) On() {
	fmt.Println("DVD播放器开启")
}
func (p *DVDPlayer) Play() {
	fmt.Println("DVD播放器开始播放")
}

type Projector struct {
}

func (p *Projector) On() {
	fmt.Println("投影仪开启")
}
func (p *Projector) SetInput() {
	fmt.Println("投影仪设为DVD模式")
}

type StereoSystem struct{}

func (s *StereoSystem) On() {
	fmt.Println("音响系统开启")
}

func (s *StereoSystem) SetMode() {
	fmt.Println("音响系统设为影院模式")
}

type HomeTheaterFacade struct {
	dvdPlayer    *DVDPlayer
	projector    *Projector
	stereoSystem *StereoSystem
}

func NewHomeTheater() *HomeTheaterFacade {
	return &HomeTheaterFacade{
		dvdPlayer:    &DVDPlayer{},
		projector:    &Projector{},
		stereoSystem: &StereoSystem{},
	}
}
func (f *HomeTheaterFacade) WatchMovie() {
	fmt.Println("准备看电影...")
	f.dvdPlayer.On()
	f.projector.On()
	f.projector.SetInput()
	f.stereoSystem.On()
	f.stereoSystem.SetMode()
	f.dvdPlayer.Play()
	fmt.Println("放映开始")
}

func main() {
	homeTheater := NewHomeTheater()
	homeTheater.WatchMovie()
}

/**
不是“增加功能”，而是“简化调用”。

外观模式不会封锁子系统的访问，只是提供更友好的入口。

外观模式经常和中介者模式、**门面服务（service layer）**配合使用。
*/
