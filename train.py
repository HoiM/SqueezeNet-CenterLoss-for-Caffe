#!/usr/bin/env python

import os
import sys
import google.protobuf as pb2
import matplotlib.pyplot as plt
CAFFE_ROOT = "../caffe-face/"
sys.path.append(CAFFE_ROOT + "python/")
import caffe
from caffe.proto import caffe_pb2
import numpy as np

class SolverWapper(object):
    
    
    def __init__(self, solver, output_dir, pretrained_model=None, gpu_id=0):
        
        self.output_dir = output_dir
        
        caffe.set_mode_gpu()
        caffe.set_device(gpu_id)
        self.solver = caffe.SGDSolver(solver)
        if pretrained_model is not None:
            print "Loading pretrained model, weights from {:s}".format(pretrained_model)
            self.solver.net.copy_from(pretrained_model)
        
        self.solver_param = caffe_pb2.SolverParameter()
        with open(solver, "rt") as f:
            pb2.text_format.Merge(f.read(), self.solver_param)

    def train_model(self):
        
        display = self.solver_param.display
        snapshot = self.solver_param.snapshot
        max_iters = self.solver_param.max_iter
 
        last_snapshot_iter = -1
        softmaxlosstxt = os.path.join(self.output_dir, 'softmax.txt')
        centerlosstxt = os.path.join(self.output_dir, 'center.txt')
        losstxt = os.path.join(self.output_dir, 'loss.txt')
        fs = open(softmaxlosstxt, 'w')
        fc = open(centerlosstxt, 'w')
        fl = open(losstxt, 'w')

        while self.solver.iter < max_iters:

            self.solver.step(1)

            softmaxloss = self.solver.net.blobs['loss'].data
            centerloss = self.solver.net.blobs['center_loss'].data
            loss = softmaxloss + centerloss*0.008

            fs.write('{} {}\n'.format(self.solver.iter - 1, softmaxloss))
            fs.flush()
            fc.write('{} {}\n'.format(self.solver.iter - 1, centerloss))
            fc.flush()
            fl.write('{} {}\n'.format(self.solver.iter - 1, loss))
            fl.flush()

        fs.close()
        fc.close()
        fl.close()


if __name__ == '__main__':
    solver = './model/solver.prototxt'
    output_dir = './output/'
    pretrained_model = None
    gpu_id = 0
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        
    sw = SolverWapper(solver, output_dir, pretrained_model, gpu_id)
    print "Solving begins ..."
    sw.train_model()
    print "Solving done ..."
    sw.solver.net.save(output_dir + 'weights.caffemodel')
    del sw
