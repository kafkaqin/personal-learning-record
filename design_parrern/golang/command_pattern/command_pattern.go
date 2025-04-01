package main

import "fmt"

//主要角色
//命令（Command）：声明执行操作的接口，通常包含一个 Execute 方法，有时还包含 Undo 方法用于支持撤销操作。
//
//具体命令（ConcreteCommand）：实现命令接口，定义与接收者之间的绑定关系，调用接收者相应的操作。
//
//接收者（Receiver）：执行与请求相关的实际操作，是命令真正作用的对象。
//
//调用者（Invoker）：负责调用命令对象执行请求。调用者可以在不同时间点调用命令，从而实现请求的延迟执行或撤销操作。

type Command interface {
	Execute()
	Undo()
}

type Light struct {
	name string
}

func (l *Light) On() {
	fmt.Println("light on", l.name)
}
func (l *Light) Off() {
	fmt.Println("light off", l.name)
}

type LightOnCommand struct {
	light *Light
}

func (l *LightOnCommand) Execute() {
	l.light.On()
}

func (l *LightOnCommand) Undo() {
	l.light.Off()
}

type LightOffCommand struct {
	light *Light
}

func (l *LightOffCommand) Execute() {
	l.light.Off()
}

func (l *LightOffCommand) Undo() {
	l.light.On()
}

type RemoteControl struct {
	command Command
}

func (r *RemoteControl) SetCommand(cmd Command) {
	r.command = cmd
}

func (r *RemoteControl) PressButton() {
	if r.command != nil {
		r.command.Execute()
	}
}
func (r *RemoteControl) PressUndo() {
	if r.command != nil {
		r.command.Undo()
	}
}

func main() {
	livingRoomLight := &Light{
		name: "living_room",
	}

	lightOnCommand := &LightOnCommand{
		light: livingRoomLight,
	}

	lightOffCommand := &LightOffCommand{
		light: livingRoomLight,
	}

	remote := &RemoteControl{}
	remote.SetCommand(lightOnCommand)

	remote.PressButton()
	remote.PressUndo()

	remote.SetCommand(lightOffCommand)

	remote.PressButton()
	remote.PressUndo()
}
