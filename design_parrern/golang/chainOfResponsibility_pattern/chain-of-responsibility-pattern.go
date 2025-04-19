package main

import "fmt"

/**
职责链设计模式（Chain of Responsibility Pattern）是一种行为型设计模式，允许多个对象都有机会处理请求
，将这些对象连成一条链，沿着这条链传递请求，直到有一个对象处理它为止。

避免请求发送者与处理者之间的耦合。
动态地组合处理链，增强系统灵活性。
支持请求的拦截、放行、过滤、终止等。
*/

type Request struct {
	Name     string
	LeaveDay int
}
type Handler interface {
	SetNext(handler Handler)
	Handle(req *Request)
}

type BaseHandler struct {
	nextHandler Handler
}

func (b *BaseHandler) SetNext(handler Handler) {
	b.nextHandler = handler
}

func (b BaseHandler) PassToNext(request *Request) {
	if b.nextHandler != nil {
		b.nextHandler.Handle(request)
	} else {
		fmt.Println("No next handler")
	}
}

type Leader struct {
	BaseHandler
}

func (l *Leader) Handle(request *Request) {
	if request.LeaveDay <= 1 {
		fmt.Printf("组长批准 %s 请假 %d 天\n", request.Name, request.LeaveDay)
	} else {
		fmt.Println("组长权限不足，向上汇报")
		l.PassToNext(request)
	}
}

type Manager struct {
	BaseHandler
}

func (m *Manager) Handle(request *Request) {
	if request.LeaveDay <= 3 {
		fmt.Printf("经理批准 %s 请假 %d 天\n", request.Name, request.LeaveDay)
	} else {
		fmt.Println("经理权限不足，向上汇报")
		m.PassToNext(request)
	}
}

type Director struct {
	BaseHandler
}

func (d *Director) Handle(request *Request) {
	fmt.Printf("总监批准 %s 请假 %d 天\n", request.Name, request.LeaveDay)
}

func main() {
	leader := &Leader{}
	manager := &Manager{}
	director := &Director{}
	leader.SetNext(manager)
	manager.SetNext(director)

	req1 := &Request{Name: "张三", LeaveDay: 1}
	req2 := &Request{Name: "李四", LeaveDay: 2}
	req3 := &Request{Name: "王五", LeaveDay: 3}
	leader.Handle(req1)
	leader.Handle(req2)
	leader.Handle(req3)
}
