package main

import "fmt"

/**
状态设计模式（State Pattern）是一种行为型设计模式，它允许一个对象在其内部状态改变时改变它的行为，
看起来就像改变了它的类一样。
状态模式（State Pattern）允许一个对象在其内部状态改变时，改变它的行为。就像对象换了一个“角色”，行为也随之改变。

 作用（为什么用状态模式）
将状态相关的行为封装到独立的状态类中，使状态切换更清晰。
避免在一个类中写大量 if/else 或 switch-case。
更易维护、扩展状态逻辑，符合开闭原则。

*/

type State interface {
	PressButton(e *Elevator)
}

type Elevator struct {
	state State
}

func (e *Elevator) SetState(state State) {
	e.state = state
}

func (e *Elevator) PressButton() {
	e.state.PressButton(e)
}

type StopState struct{}

func (s *StopState) PressButton(e *Elevator) {
	fmt.Println("电梯开始运行....")
	e.SetState(&RunningState{})
}

type RunningState struct{}

func (s *RunningState) PressButton(e *Elevator) {
	fmt.Println("电梯正在运行，忽略按钮...")
}

func main() {
	//e := &Elevator{}
	//e.SetState(&StopState{})
	//e.PressButton()
	//e.PressButton()
	//e.PressButton()
	//e.SetState(&StopState{})
	//e.PressButton()

	order := NewOrder()

	order.Pay()     // 支付成功
	order.Ship()    // 订单已发货
	order.Deliver() // 收货完成
	order.Cancel()  // 收货完成
	order.Pay()     // 已完成

}

// ========== 状态接口 ==========
type OrderState interface {
	Pay(o *Order)
	Ship(o *Order)
	Deliver(o *Order)
	Cancel(o *Order)
	GetName() string
}

// ========== 订单 ==========
type Order struct {
	State OrderState
}

func NewOrder() *Order {
	return &Order{State: &PendingState{}}
}

func (o *Order) SetState(state OrderState) {
	fmt.Printf("订单状态变更：%s → %s\n", o.State.GetName(), state.GetName())
	o.State = state
}

func (o *Order) Pay()     { o.State.Pay(o) }
func (o *Order) Ship()    { o.State.Ship(o) }
func (o *Order) Deliver() { o.State.Deliver(o) }
func (o *Order) Cancel()  { o.State.Cancel(o) }

// ========== 状态实现 ==========

type PendingState struct{}

func (s *PendingState) Pay(o *Order) {
	fmt.Println("支付成功")
	o.SetState(&PaidState{})
}
func (s *PendingState) Ship(o *Order) {
	fmt.Println("未付款，无法发货")
}
func (s *PendingState) Deliver(o *Order) {
	fmt.Println("未付款，无法收货")
}
func (s *PendingState) Cancel(o *Order) {
	fmt.Println("订单已取消")
	o.SetState(&CancelledState{})
}
func (s *PendingState) GetName() string { return "待付款" }

type PaidState struct{}

func (s *PaidState) Pay(o *Order) {
	fmt.Println("已付款，无需重复支付")
}
func (s *PaidState) Ship(o *Order) {
	fmt.Println("订单已发货")
	o.SetState(&ShippedState{})
}
func (s *PaidState) Deliver(o *Order) {
	fmt.Println("未发货，无法收货")
}
func (s *PaidState) Cancel(o *Order) {
	fmt.Println("订单无法取消，已付款")
}
func (s *PaidState) GetName() string { return "已付款" }

type ShippedState struct{}

func (s *ShippedState) Pay(o *Order)  { fmt.Println("已付款") }
func (s *ShippedState) Ship(o *Order) { fmt.Println("已发货") }
func (s *ShippedState) Deliver(o *Order) {
	fmt.Println("收货完成")
	o.SetState(&CompletedState{})
}
func (s *ShippedState) Cancel(o *Order) {
	fmt.Println("已发货，不能取消订单")
}
func (s *ShippedState) GetName() string { return "已发货" }

type CompletedState struct{}

func (s *CompletedState) Pay(o *Order)     { fmt.Println("订单已完成") }
func (s *CompletedState) Ship(o *Order)    { fmt.Println("订单已完成") }
func (s *CompletedState) Deliver(o *Order) { fmt.Println("订单已完成") }
func (s *CompletedState) Cancel(o *Order)  { fmt.Println("订单已完成，不能取消") }
func (s *CompletedState) GetName() string  { return "已完成" }

type CancelledState struct{}

func (s *CancelledState) Pay(o *Order)     { fmt.Println("订单已取消，无法支付") }
func (s *CancelledState) Ship(o *Order)    { fmt.Println("订单已取消，无法发货") }
func (s *CancelledState) Deliver(o *Order) { fmt.Println("订单已取消，无法收货") }
func (s *CancelledState) Cancel(o *Order)  { fmt.Println("订单已取消") }
func (s *CancelledState) GetName() string  { return "已取消" }

// ======
// type OrderStatus int
//
// const (
//
//	StatusPending OrderStatus = iota + 1
//	StatusPaid
//	StatusShipped
//	StatusCompleted
//	StatusCancelled
//
// )
//
//	type Order struct {
//		Status OrderStatus
//		State  OrderState
//	}
//
//	func NewOrder(status OrderStatus) *Order {
//		o := &Order{Status: status}
//		o.State = GetStateByStatus(status)
//		return o
//	}
//
//	func (o *Order) SetState(state OrderState, status OrderStatus) {
//		fmt.Printf("订单状态变更：%s → %s\n", o.State.GetName(), state.GetName())
//		o.State = state
//		o.Status = status
//	}
//
// func (o *Order) Pay()     { o.State.Pay(o) }
// func (o *Order) Ship()    { o.State.Ship(o) }
// func (o *Order) Deliver() { o.State.Deliver(o) }
// func (o *Order) Cancel()  { o.State.Cancel(o) }
//
//	func GetStateByStatus(status OrderStatus) OrderState {
//		switch status {
//		case StatusPending:
//			return &PendingState{}
//		case StatusPaid:
//			return &PaidState{}
//		case StatusShipped:
//			return &ShippedState{}
//		case StatusCompleted:
//			return &CompletedState{}
//		case StatusCancelled:
//			return &CancelledState{}
//		default:
//			panic("未知状态")
//		}
//	}
//func (s *PendingState) Pay(o *Order) {
//	fmt.Println("支付成功")
//	o.SetState(&PaidState{}, StatusPaid)
//}
//// 模拟从数据库读取
//orderFromDB := NewOrder(StatusPaid)
//orderFromDB.Ship()
//orderFromDB.Deliver()
