import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

class TRTWrapper():
    """TensorRT model wrapper."""
    def __init__(self, model_path, batch):
        self.model_path = model_path
        self._batch = batch
        self._bindings = None

    @property
    def batch(self):
        return self._batch

    @batch.setter
    def batch(self, value):
        self._batch = value

    @property
    def bindings(self):
        return self._bindings

    @bindings.setter
    def bindings(self, value):
        self._bindings = value

    def load_model(self):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER) # serialized ICudEngine을 deserialized하기 위한 클래스 객체
        trt.init_libnvinfer_plugins(None, "") # plugin 사용을 위함
        with open(self.model_path, 'rb') as f:
            self.engine = runtime.deserialize_cuda_engine(f.read()) # trt 모델을 읽어 serialized ICudEngine을 deserialized함
        
        self.context = self.engine.create_execution_context() # ICudEngine을 이용해 inference를 실행하기 위한 context class생        assert self.engine 
        assert self.context
        
        self.alloc_buf()

    def inference(self, input_image):
        image = input_image.transpose(0, 3, 1, 2) # NHWC to NWHC
        image = np.ascontiguousarray(image) 
        cuda.memcpy_htod(self.inputs[0]['allocation'], image) # input image array(host)를 GPU(device)로 보내주는 작업
        self.context.execute_v2(self.allocations) #inference 실행!
        for o in range(len(self.outputs)):
            cuda.memcpy_dtoh(self.outputs[o]['host_allocation'], self.outputs[o]['allocation']) # GPU에서 작업한 값을 host로 보냄
        
        detections = self.outputs[0]['host_allocation'] # detection된 object개수
        return detections


    def alloc_buf(self):
        # Setup I/O bindings
        self.inputs = []
        self.outputs = []
        self.allocations = []

        print(self.engine.num_bindings)
        for i in range(self.engine.num_bindings):
            is_input = False
            if self.engine.binding_is_input(i): # i번째 binding이 input인지 확인
                is_input = True 
            name = self.engine.get_binding_name(i) # i번째 binding의 이름
            dtype = np.dtype(trt.nptype(self.engine.get_binding_dtype(i))) # i번째 binding의 data type
            shape = self.context.get_binding_shape(i) # i번째 binding의 shape

            if is_input and shape[0] < 0:
                assert self.engine.num_optimization_profiles > 0
                profile_shape = self.engine.get_profile_shape(0, name)
                assert len(profile_shape) == 3  # min,opt,max
                # Set the *max* profile as binding shape
                self.context.set_binding_shape(i, profile_shape[2])
                shape = self.context.get_binding_shape(i)
            if is_input:
                self.batch_size = shape[0]
            size = dtype.itemsize # data type의 bit수
            for s in shape:
                size *= s # data type * 각 shape(e.g input의 경우 [1,3,640,640]) element 을 곱하여 size에 할당

            allocation = cuda.mem_alloc(size) # 해당 size만큼의 GPU memory allocation함
            host_allocation = None if is_input else np.zeros(shape, dtype)
            binding = {
                "index": i,
                "name": name,
                "dtype": dtype,
                "shape": list(shape),
                "allocation": allocation,
                "host_allocation": host_allocation,
            }
            self.allocations.append(allocation)
            if self.engine.binding_is_input(i): # binding이 input이면
                self.inputs.append(binding)
            else: # 아니면 binding은 모두 output임
                self.outputs.append(binding)
            print("{} '{}' with shape {} and dtype {}".format(
                "Input" if is_input else "Output",
                binding['name'], binding['shape'], binding['dtype']))        

        assert self.batch_size > 0
        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        assert len(self.allocations) > 0

    def input_spec(self):
        return self.inputs[0]['shape'], self.inputs[0]['dtype']

    def output_spec(self):
        specs = []
        for o in self.outputs:
            specs.append((o['shape'], o['dtype']))
        return specs

def create_model_wrapper(model_path: str, batch_size: int, device: str):
    """Create model wrapper class."""
    assert trt and cuda, f"Loading TensorRT, Pycuda lib failed."
    model_wrapper = TRTWrapper(model_path, batch_size)

    return model_wrapper