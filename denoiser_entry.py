import torch
import os
import subprocess
import torch.utils.cpp_extension as cpp_extension
import sys
cur_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(cur_dir)

from trainlive import init_training_state, train_and_eval

sys.stdout = open("CONOUT$", 'w')
sys.stderr = open("CONOUT$", 'w')

def definitely_available():
    return True
cpp_extension.verify_ninja_availability = definitely_available

vs_magic_path = "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\Community\VC\\Auxiliary\\Build\\vcvars64.bat"
cpp_path = os.path.join(cur_dir, "cpp_extensions")

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

#RGBA to RGB + permute
def convert_for_pytorch(tensor):
    return tensor[..., 0:3].permute(0, 3, 1, 2)

def denoiser_train_and_eval(training_state, inputs, output):
    color, ref_color, normal, albedo = inputs

    color = convert_for_pytorch(color)
    ref_color = convert_for_pytorch(ref_color)
    normal = convert_for_pytorch(normal)
    albedo = convert_for_pytorch(albedo)

    pytorch_output = train_and_eval(training_state, color, ref_color, normal, albedo)
    output[..., 3] = 1
    output[..., 0:3] = pytorch_output.permute(0, 2, 3, 1)

def denoiser_entry(input_ready, output_ready, inputs, output, buffer_sizes):
    inputs, output = falcor_bindings.bind_buffers(inputs, output, buffer_sizes)
    training_state = init_training_state()
    while True:
        input_ready.acquire()

        denoiser_train_and_eval(training_state, inputs, output)

        torch.cuda.synchronize()

        output_ready.release()
