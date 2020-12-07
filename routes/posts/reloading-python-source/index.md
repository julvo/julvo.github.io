---
template: post.html
title: Reloading Python source code of a running programme without losing state
img: https://github.com/julvo/reloading/raw/master/examples/demo/demo.gif
width: 3
tags:
    - python
---

Here could be a paragraph with lots of interesting content

![demo](https://github.com/julvo/reloading/raw/master/examples/demo/demo.gif)

```python
from reloading import reloading

for i in reloading(range(100)):
    print(i)

```