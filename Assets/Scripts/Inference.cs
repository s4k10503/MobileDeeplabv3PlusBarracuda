using Unity.Barracuda;
using UnityEngine;
using System;
using System.Collections.Generic;
using System.Linq;

namespace MobileDeeplabv3Plus
{
    public class Inference : IDisposable
    {
        public RenderTexture SegmentationTexture { get; private set; }

        private IWorker _worker;
        private ComputeBuffer _inputBuffer;
        private ComputeBuffer _outputBuffer;
        private ResourceSet _resources;
        private Utils.ThreadSize _threadSize;

        private static readonly int s_inputWidth = 256;
        private static readonly int s_inputHeight = 256;
        private static readonly int s_inputChannels = 3;

        public Inference(ResourceSet resourceSet)
        {
            Initialize(resourceSet);
        }

        private void Initialize(ResourceSet resourceSet)
        {
            _resources = resourceSet;

            var model = ModelLoader.Load(_resources.Model);
            _worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, model);

            _inputBuffer = new ComputeBuffer(1 * s_inputWidth * s_inputHeight * s_inputChannels, sizeof(float));
            _outputBuffer = new ComputeBuffer(1 * s_inputWidth * s_inputHeight, sizeof(int));

            SegmentationTexture = Utils.CreateRenderTexture(s_inputWidth, s_inputHeight);
        }

        private Utils.ThreadSize GetThreadSize(ComputeShader shader)
        {
            shader.GetKernelThreadGroupSizes(0, out uint x, out uint y, out uint z);
            return new Utils.ThreadSize(x, y, z);
        }

        public void ProcessImage(RenderTexture sourceTexture)
        {
            PreprocessImage(sourceTexture);
            PerformInference();
            PostprocessImage(sourceTexture);
        }

        private void PreprocessImage(RenderTexture sourceTexture)
        {
            var preprocess = _resources.Preprocess;
            _threadSize = GetThreadSize(preprocess);
            preprocess.SetTexture(0, "InputTexture", sourceTexture);
            preprocess.SetBuffer(0, "OutputTensor", _inputBuffer);
            preprocess.SetInt("Size", s_inputWidth);
            preprocess.Dispatch(0, s_inputWidth / (int)_threadSize.X, s_inputHeight / (int)_threadSize.Y, (int)_threadSize.Z);
        }

        private void PerformInference()
        {
            using (var tensor = new Tensor(1, s_inputWidth, s_inputHeight, s_inputChannels, _inputBuffer))
            {
                _worker.Execute(tensor);
            }

            var output = _worker.PeekOutput();
            _outputBuffer.SetData(output.AsInts());
        }

        private void PostprocessImage(RenderTexture sourceTexture)
        {
            var postprocess = _resources.Postprocess;
            _threadSize = GetThreadSize(postprocess);
            postprocess.SetBuffer(0, "InputTensor", _outputBuffer);
            postprocess.SetTexture(0, "InputTexture", sourceTexture);
            postprocess.SetTexture(0, "OutputTexture", SegmentationTexture);
            postprocess.SetInt("Size", s_inputWidth);
            postprocess.Dispatch(0, s_inputWidth / (int)_threadSize.X, s_inputHeight / (int)_threadSize.Y, (int)_threadSize.Z);
        }

        public void Dispose()
        {
            _worker.Dispose();
            _inputBuffer.Dispose();
            _outputBuffer.Dispose();

            if (SegmentationTexture != null)
            {
                SegmentationTexture.Release();
                SegmentationTexture = null;
            }
        }
    }
}
