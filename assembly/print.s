
.print:
  push ebp
  mov ebp,esp
  sub esp,0x20
  mov [ebp - 8],10
  mov eax,[ebp - 8]
  cmp eax, 0
  je .end_1
  mov [ebp - 8], 100
.end_1:
   mov [ebp - 8], 10000
   jmp .end
.end:
  leave
     mov esp,ebp
     pop  ebp
  ret