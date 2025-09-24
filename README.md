***ROUGH AMOUNT OF TIME TAKEN --> like 3-4 weeks, or (6-7)/2 weeks***

**IMPORTANT NOTICES**

This code _REQUIRES_ shape_predictor_68_face_landmarks.dat for it to work

You ABSOLUTELY need this for the face tracker to work. *-yes,i have a version that uses a, d, w, s, and g, f -yes,i accidently deleted it*

**How this works:**

I used:

```python
def projectTo2D(x, y, z, centerX, centerY, depth=20):
    scale = depth / (depth + z)
    screenX = round(x * scale + centerX)
    screenY = round(y * scale + centerY)
```
to convert the points into 2D.

But first, I had to get a cube in 3D. for that, I defined 3D coordinates using self-typed :(
pile of code in which I was deeply annoyed that I had to do and could not be proceederly generated easily

```python
cubeCorners = [
    (-size, -size, -size), (size, -size, -size), #back corners 
    (size, size, -size), (-size, size, -size),
    (-size, -size, size), (size, -size, size),    #front corners
    (size, size, size), (-size, size, size)
]
```

I'll fill in the rest of this later. I got to do my homework.

It should be posted :P

for Hack Club, Perse school
