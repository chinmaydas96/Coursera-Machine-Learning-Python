from urllib import urlencode
from urllib2 import urlopen
from json import loads, dumps
from collections import OrderedDict
import numpy as np
import os



class Submission():

    def __init__(self, homework, part_names, srcs, output):
        self.__homework = homework
        self.__part_names = part_names
        self.__srcs = srcs
        self.__output = output
        self.__submit_url = 'https://www-origin.coursera.org/api/onDemandProgrammingImmediateFormSubmissions.v1'
        self.__login = None
        self.__password = None

    def submit(self):
        print '==\n== Submitting Solutions | Programming Exercise %s\n==' % self.__homework
        self.login_prompt()

        parts = OrderedDict()
        for part_id, _ in enumerate(self.__srcs,1):
            parts[str(part_id)] = {'output': self.__output(part_id)}

        result, response = self.request(parts)
        response = loads(response)
        try:
            print response['errorMessage']
            return
        except:
            pass
        print '=='
        print '== %43s | %9s | %-s' % ('Part Name', 'Score', 'Feedback')
        print '== %43s | %9s | %-s' % ('---------', '-----', '--------')
        

        for part in parts:
            partFeedback = response['partFeedbacks'][part]
            partEvaluation = response['partEvaluations'][part]
            score = '%d / %3d' % (partEvaluation['score'], partEvaluation['maxScore'])
            print '== %43s | %9s | %-s' % (self.__part_names[int(part)-1], score, partFeedback)

        evaluation = response['evaluation']
    

        totalScore = '%d / %d' % (evaluation['score'], evaluation['maxScore'])
        print '==                                   --------------------------------'
        print '== %43s | %9s | %-s\n' % (' ', totalScore, ' ')
        print '=='

        if not os.path.isfile('token.txt'):
            with open('token.txt', 'w') as f:
                f.write(self.__login + '\n')
                f.writelines(self.__password)


    def login_prompt(self):
        try:
            with open('token.txt', 'r') as f:
                self.__login = f.readline().strip()
                self.__password = f.readline().strip()
        except IOError:
            pass

        if self.__login is not None and self.__password is not None:
            reenter = raw_input('Use token from last successful submission (%s)? (Y/n): ' % self.__login)

            if reenter == '' or reenter[0] == 'Y' or reenter[0] == 'y':
                return

        if os.path.isfile('token.txt'):
            os.remove('token.txt')
        self.__login = raw_input('Login (email address): ')
        self.__password = raw_input('Token: ')

    def request(self, parts):

        params = {
            'assignmentSlug': self.__homework,
            'secret': self.__password,
            'parts': parts,
            'submitterEmail': self.__login}

        params = urlencode({'jsonBody': dumps(params)})
        f = urlopen(self.__submit_url, params)
        try:
            return 0, f.read()
        finally:
            f.close()

def sprintf(fmt, arg):
    "emulates (part of) Octave sprintf function"
    if isinstance(arg, tuple):
        # for multiple return values, only use the first one
        arg = arg[0]

    if isinstance(arg, (np.ndarray, list)):
        # concatenates all elements, column by column
        return ' '.join(fmt % e for e in np.asarray(arg).ravel('F'))
    else:
        return fmt % arg