import os
import pypandoc
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
from kivy.uix.codeinput import CodeInput
from kivy.uix.label import Label
from kivy.uix.rst import RstDocument
from kivy.uix.textinput import TextInput
from kivy.uix.spinner import Spinner
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.splitter import Splitter
from kivy.uix.scrollview import ScrollView
from kivy.uix.popup import Popup
from kivy.clock import Clock
from functools import partial
from kivy.animation import Animation
from Submission import Submission


class resourceHandler():
    def __init__(self, **args):
        pass


    def read_token(self,instance):
        path = '../'+instance.current_ex+'/token.txt'
        try:
            credentials = open(path)
            instance.email = credentials.readline().strip()
            instance.token = credentials.readline().strip()
            return True
        except Exception, e:
            return False



    def files(self,excercise):
        path = 'res/'+excercise+'/sources.txt'
        filehandler = open(path)
        filelist=[]
        while True:
            try:
                filelist.append(filehandler.next())
            except Exception, e:
                return filelist
    def manual(self, excercise):
        path = 'res/'+excercise+'/manual.md'
        return pypandoc.convert(path,'rst')

    def writeFile(self,excercise,filename):
        path = '../'+excercise+'/'+filename
        f=open(path,'w')
        return f

    def readFile(self,excercise,filename):
        path = '../'+excercise+'/'+filename
        #print 'Opening ',path
        f=open(path,'r')
        return f.read()

class MainScreen(BoxLayout):

    def __init__(self, welcome=False, **kwargs):
        super(MainScreen, self).__init__(**kwargs)
        self.orientation='vertical'
        self.current_ex = 'ex1'
        self.current_file = 'warmUpExercise.py'
        self.submit_ob = Submission()
        self.element=resourceHandler()

        if welcome:
            welcome_popup = Popup(title='Coursera ML in Python', content=Label(text='Hello World'),size_hint=(1, 1))
            self.add_widget(welcome_popup)
            welcome_popup.open()
            Clock.schedule_once(self.start_app,3)
        else:
            self.bind(size=self.draw_screen)
  
    def start_app(self,*args):
        self.draw_screen()
        self.bind(size=self.draw_screen)

    def draw_screen(self,*args):
        self.clear_widgets()
        self.add_widget(self.titlebar())
        self.add_widget(self.maineditor())
        scrollbar = ScrollView(size_hint=(1,None))
        #TODO: Make filebar scrollable for smaller screens and many files
        #scrollbar.add_widget(self.filebar())
        #self.add_widget(scrollbar)
        self.add_widget(self.filebar())
        self.add_widget(self.console())

    def titlebar(self):
        layout=BoxLayout(padding='2sp',size_hint=(1,None),height='65sp')
        layout.orientation='horizontal'

        #credentials = self.accept_credentials()
        self.submit_popup = Popup(title='Enter credentials',content=self.accept_credentials(),size_hint=(0.6, 0.35))
        #credentials.children[1].bind(on_press=self.submit_popup.dismiss)

        submit = Button(text='Submit',size_hint=(0.4,1))
        if self.element.read_token(self):
            submit.bind(on_press=partial(self.submit_assignment))
        else:
            submit.bind(on_press=self.submit_popup.open)

        run = Button(text='Run',size_hint=(0.4,1))
        run.bind(on_press=self.run)

        ex_dropdown = Spinner(text=self.current_ex,size_hint=(1,1))
        ex_dropdown.values = os.listdir('./res/')
        ex_dropdown.bind(text=self.updateExercise)

        layout.add_widget(run)
        layout.add_widget(ex_dropdown)
        layout.add_widget(submit)

        return layout


    def console(self):
        layout = FloatLayout(size_hint=(1,None),height=100)
        self.info_label = TextInput(size_hint=(1,None),readonly=True,background_color=(0,0,0,1),foreground_color=(1,1,1,1),opacity=0)
        self.info_label.text_size = self.size
        self.info_label.text = 'console'
        self.info_label.height = '150pt'
        self.info_label.top = 0
        layout.add_widget(self.info_label)
        return layout


    def accept_credentials(self):
        main_layout= BoxLayout(padding='2sp')
        main_layout.orientation='vertical'
        layout=GridLayout(padding='2sp')
        layout.cols=2
        layout.add_widget(Label(text='Email id:'))
        email = TextInput(multiline=False)
        layout.add_widget(email)
        token = TextInput(multiline=False)
        layout.add_widget(Label(text='Submission Token:'))
        layout.add_widget(token)
        main_layout.add_widget(layout)
        submit = Button(text='Submit',size_hint=(1,0.4))
        submit.bind(on_press=partial(self.submit_assignment,email,token))
        main_layout.add_widget(submit)
        return main_layout

    def submit_assignment(self,*largs):
        #Make submit_ob local if not used anywhere else 
        if len(largs)>1:
            self.submit_ob.__login = largs[0].text
            self.submit_ob.__password = largs[1].text
        else:
            self.submit_ob.__login=self.email
            self.submit_ob.__password=self.token

        print 'Email',self.submit_ob.__login
        print 'Token', self.submit_ob.__password
        self.submit_popup.dismiss()
        #TODO:submission call
        #self.show_error(self.submit_ob.submit())


    def updateExercise(self,spinner,text):
        self.current_ex=text
        current_file = self.element.files(self.current_ex)[0]
        if current_file.endswith('\n'):
            current_file=current_file[:-1]
        self.current_file= current_file
        self.draw_screen()
        print 'Current Exercise changed to: ', self.current_ex



    def run(self,instance):
        #TODO: Display output in popup
        self.show_error('Cannot run')
        print('The button <%s> is being pressed' % instance.text)

    
   
        
    def maineditor(self):
        layout=BoxLayout()
        if self.width < self.height:
            layout.orientation='vertical'
        else:
            layout.orientation='horizontal'
        #self.bind(self.current_ex=self.update_currentFile)
        man = self.element.manual(self.current_ex)
        codeFile = self.element.readFile(self.current_ex,self.current_file)
        code = CodeInput(text=codeFile)
        code.bind(focus =self.schedule_reload)
        splitter = Splitter()
        if layout.orientation == 'vertical':
            splitter.sizable_from='bottom'
        else:
            splitter.sizable_from='right'
        splitter.add_widget(code)
        layout.add_widget(splitter)

        layout.add_widget(RstDocument(text=man))
        return layout

    def saveAssignment(self,assignment,*largs):
        print 'callback called'
        try:
            if not self.element.readFile(self.current_ex,self.current_file)==assignment.text:
                filehandler = self.element.writeFile(self.current_ex,self.current_file)
                filehandler.write(assignment.text)
                print 'INFO: Autosaved file'
        except Exception, e:
            raise e
            self.show_error(e)


    def schedule_reload(self,instance,value):
        if value:
            #Schedule Update
            self.callback = partial(self.saveAssignment,instance)
            Clock.schedule_interval(self.callback,5)
        else:
            #TODO:When clicking on another file, both focus=False and filebar button callbacks are executed simultaneously leading to deadlock
            Clock.unschedule(self.callback)
            #self.saveAssignment(instance)
            #Update now
            
    
    def filebar(self):
        layout=BoxLayout(padding='2sp',size_hint=(1,None),height='100sp')
        layout.orientation='horizontal'
        files = self.element.files(self.current_ex)
        for f in files:
            if f.strip() == self.current_file:
                button = ToggleButton(text=f,group = self.current_ex,state='down')
            else:
                button = ToggleButton(text=f,group = self.current_ex,state='normal')
            button.bind(on_press=self.update_currentFile)
            layout.add_widget(button)
        return layout

    def update_currentFile(self,instance):
        if instance.text.endswith('\n'):
            instance.text=instance.text[:-1]
        self.current_file = instance.text
        self.draw_screen()
        print 'Current file changed to: ', self.current_file

    def show_error(self, e):
        self.info_label.text = str(e)
        duration = len(self.info_label.text)/10
        anim = Animation(top=190.0, opacity=1, d=0.5) +\
            Animation(top=190.0, d=duration) +\
            Animation(top=0, opacity=0, d=2)        
        anim.start(self.info_label)



class CourseraApp(App):

    def build(self):
        return MainScreen(welcome=True)
    def on_pause(self):
        return True


if __name__ == '__main__':	
    CourseraApp().run()
