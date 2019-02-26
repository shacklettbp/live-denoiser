import torch
import os
import subprocess
import torch.utils.cpp_extension as cpp_extension
import sys
sys.stdout = open(os.devnull, 'w')
sys.stderr = open(os.devnull, 'w')

def definitely_available():
    return True
cpp_extension.verify_ninja_availability = definitely_available

vs_magic_path = "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\Community\VC\\Auxiliary\\Build\\vcvars64.bat"
cpp_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "cpp_extensions")

orig_subprocess_run = subprocess.run
def special_subprocess_run(*args, **kwargs):
    args = list(args)
    args[0] = [vs_magic_path, '>nul', '&&'] + args[0]
    return orig_subprocess_run(*args, **kwargs, shell=True)
subprocess.run = special_subprocess_run

def get_cpp_path(filename):
    return os.path.join(cpp_path, filename)

os.environ["PATH"] += os.pathsep + cpp_path # Ninja has to be here
#extra LD flags because sys.executable path is not the python interpreter, it is the Falcor program
falcor_bindings = cpp_extension.load(name='falcor_bindings', sources=[get_cpp_path('bind_buffers.cpp'), get_cpp_path('bind_buffers.cu')], extra_ldflags=['/LIBPATH:{}'.format(os.path.join(os.path.dirname(sys.executable), os.path.join("Python", "libs")))])

subprocess.run = orig_subprocess_run

def denoiser_train_and_eval(inputs, output):
    output[..., 0].copy_(inputs[0][..., 0])

def denoiser_entry(input_ready, output_ready, inputs, output, buffer_sizes):
    inputs, output = falcor_bindings.bind_buffers(inputs, output, buffer_sizes)
    while True:
        input_ready.acquire()

        denoiser_train_and_eval(inputs, output)

        torch.cuda.synchronize()

        output_ready.release()
