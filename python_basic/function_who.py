def computepay(h,r):
    if h>40:
        pay=((h-40)*r*1.5)+(40*r)
    else:
        pay=h*r


    return pay


hrs = input("Enter Hours:")
rate = input("Enter rate:")
h = float(hrs)
r = float(rate)

p = computepay(h,r)
print("Pay",p)
