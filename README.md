# TorchHandy

## Introduction
This is a handy implementation of some useful pytorch modules and functions, which can be helpful for both students and researchers, especially in simplifying 
repeating coding procedure.

It's worth mentioning that most of the wrapped modules are written in a way the author is familiar with (currently), which means that some modules may be hard to use for some user, which the author should apologize in advance.

### Config
Some modules requires a "config", which is actually a python class that has the corresponding attributes. This README will continue to update the details about the attributes.  

The simplest way to use config is adding a python class called Config, and instantiate it into an object named "config", then pass this object as a parameter to the module and everything will be fine. (Probably)

An example of how to write "config":

```python
    class Config(object):
        '''
            Put the attributes you'd like to specified here
        '''
        dropout = 0.1
    
    config = Config()
    module = Module(config) # Module is a module in torchhandy that requires a "config" as a parameter. 
```

### Error Checking and Debugging
Please note that for the author's convenience almost no error checking is made and there're only few failure reports, which means that if you made some wrong configurations, you may come up with some really weird errors and have to read the f**king source code to solve them. Again, the author apologizes for this and this will be improved (sooner or later).

But there's not that much to worry about - the code written is so simple that you can easily understand what the author is doing. So do not hesitate to read or even modify the code for your covenience. The author sincerely believe that most users (if any but the author himself) has a better coding skill than the author.




