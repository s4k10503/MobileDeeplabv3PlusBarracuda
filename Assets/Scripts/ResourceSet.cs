using UnityEngine;
using Unity.Barracuda;


[CreateAssetMenu(fileName = "MobileDeeplabv3Plus",
                menuName = "ScriptableObjects/MobileDeeplabv3Plus Resource Set")]
public sealed class ResourceSet : ScriptableObject
{
    public NNModel Model;
    public ComputeShader Preprocess;
    public ComputeShader Postprocess;
}