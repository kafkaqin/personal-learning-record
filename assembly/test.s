push ebp
mov ebp, esp

mov eax , ecx  # eax = ecx
and eax, edx


leave
ret 4 + 2

####

push ebp
mov ebp, esp

push 4
push 3
mov edx, 2
mov ecs, 1
call get


leave
ret 4 + 2