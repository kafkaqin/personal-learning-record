在 Go 语言中，**参数传递是值传递**，但需要理解的是，Go 的值传递机制对于不同类型的变量有不同的表现。以下是详细的解释：

---

### **1. 基本概念：值传递与引用传递**

#### (1) **值传递**
- 在值传递中，函数接收的是调用者提供的值的副本。
- 对副本的修改不会影响原始值。

#### (2) **引用传递**
- 在引用传递中，函数接收的是调用者提供的值的内存地址（引用）。
- 对引用的修改会直接影响原始值。

---

### **2. Go 中的参数传递机制**

Go 是一种**值传递的语言**，这意味着当你将一个变量传递给函数时，实际上是将该变量的副本传递给了函数。然而，由于 Go 支持指针类型和一些复杂的数据结构（如切片、映射等），其行为可能会让人误以为是引用传递。

#### (1) **基本数据类型**
对于基本数据类型（如 `int`、`float64`、`string` 等），Go 的值传递是非常直观的：
- 函数接收到的是变量的副本。
- 修改副本不会影响原始变量。

**示例代码：**
```go
package main

import "fmt"

func modifyValue(x int) {
    x = 100 // 修改副本
}

func main() {
    num := 42
    fmt.Println("Before:", num) // 输出: Before: 42
    modifyValue(num)
    fmt.Println("After:", num)  // 输出: After: 42
}
```

**解释**：
- `modifyValue` 函数接收到的是 `num` 的副本。
- 修改副本不会影响原始变量 `num`。

---

#### (2) **复合数据类型**

对于复合数据类型（如结构体、数组、切片、映射等），情况稍微复杂一些，因为它们的行为取决于底层实现。

##### a. **结构体**
- 结构体是值类型，默认情况下也是按值传递。
- 如果需要按引用传递，可以传递结构体的指针。

**示例代码：**
```go
package main

import "fmt"

type Person struct {
    Name string
    Age  int
}

func modifyStruct(p Person) {
    p.Name = "Alice" // 修改副本
}

func modifyStructPtr(p *Person) {
    p.Name = "Bob" // 修改指针指向的值
}

func main() {
    person := Person{Name: "Charlie", Age: 30}
    fmt.Println("Before:", person) // 输出: Before: {Charlie 30}
    modifyStruct(person)
    fmt.Println("After modifyStruct:", person) // 输出: After modifyStruct: {Charlie 30}

    modifyStructPtr(&person)
    fmt.Println("After modifyStructPtr:", person) // 输出: After modifyStructPtr: {Bob 30}
}
```

**解释**：
- `modifyStruct` 接收到的是结构体的副本，修改副本不会影响原始结构体。
- `modifyStructPtr` 接收到的是结构体的指针，修改指针指向的值会影响原始结构体。

---

##### b. **数组**
- 数组是值类型，默认情况下也是按值传递。
- 如果需要按引用传递，可以传递数组的指针。

**示例代码：**
```go
package main

import "fmt"

func modifyArray(arr [3]int) {
    arr[0] = 100 // 修改副本
}

func main() {
    array := [3]int{1, 2, 3}
    fmt.Println("Before:", array) // 输出: Before: [1 2 3]
    modifyArray(array)
    fmt.Println("After:", array)  // 输出: After: [1 2 3]
}
```

**解释**：
- `modifyArray` 接收到的是数组的副本，修改副本不会影响原始数组。

---

##### c. **切片**
- 切片是引用类型，虽然它是值传递的，但它内部包含对底层数组的引用。
- 因此，对切片内容的修改会影响原始切片。

**示例代码：**
```go
package main

import "fmt"

func modifySlice(slice []int) {
    slice[0] = 100 // 修改底层数组的内容
}

func main() {
    slice := []int{1, 2, 3}
    fmt.Println("Before:", slice) // 输出: Before: [1 2 3]
    modifySlice(slice)
    fmt.Println("After:", slice)  // 输出: After: [100 2 3]
}
```

**解释**：
- 切片是值传递的，但它的值包含对底层数组的引用。
- 修改底层数组的内容会影响原始切片。

---

##### d. **映射（Map）**
- 映射是引用类型，虽然它是值传递的，但它内部包含对底层数据结构的引用。
- 因此，对映射内容的修改会影响原始映射。

**示例代码：**
```go
package main

import "fmt"

func modifyMap(m map[string]int) {
    m["key"] = 100 // 修改底层数据结构的内容
}

func main() {
    m := map[string]int{"key": 42}
    fmt.Println("Before:", m) // 输出: Before: map[key:42]
    modifyMap(m)
    fmt.Println("After:", m)  // 输出: After: map[key:100]
}
```

**解释**：
- 映射是值传递的，但它的值包含对底层数据结构的引用。
- 修改底层数据结构的内容会影响原始映射。

---

### **3. 总结**

- **Go 是值传递的语言**，所有参数传递都是按值进行的。
- **基本数据类型**（如 `int`、`float64` 等）：传递的是值的副本，修改副本不会影响原始值。
- **复合数据类型**（如结构体、数组）：默认按值传递，但如果需要按引用传递，可以传递指针。
- **引用类型**（如切片、映射）：虽然它们是值传递的，但它们的值包含对底层数据结构的引用，因此对内容的修改会影响原始变量。

通过理解这些机制，你可以更好地设计和优化你的 Go 程序。