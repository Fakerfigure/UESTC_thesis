# 有时候可能安装了多个版本的typst(homebrew安装 / cargo安装 / scoop安装 ...) 需要指定使用哪个版本
# 例如
# typst := /opt/homebrew/bin/typst
# typst := ~/.cargo/bin/typst
typst := typst

# 输出文件名
file_name := "毕业论文.pdf"

.PHONY: all build

all: build

build:
	${typst} c main.typ ${file_name} --font-path ./uestc-thesis-template/fonts --root .

watch:
	${typst} w main.typ ${file_name} --font-path ./uestc-thesis-template/fonts --root .

clean:
	rm -rf output/* *.pdf