---
layout: post
title:  "How to post images in Jekyll markdown"
date:   2022-09-17 00:00:00 -0400
categories: Tips
---
# How to post images in Jekyll markdown

I had trouble including a png plot in my first Jekyll post, so I'll post the resolution steps here as a hello-world post and to save others the headache:

1. Make a directory `assets` at the root of your github.io directory.
2. Put the desired image into the `assets` directory.
3. Reference the image with standard markdown syntax, but an absolute path starting with `/assets` will resolve correctly:
```
![plot_P_2_stable](/assets/images/output_3_0.png)
```
4. Use `jekyll serve` to render the page and ensure that the image loads correctly:
![plot_P_2_stable](/assets/images/output_3_0.png)

