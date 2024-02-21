using Oculus.Interaction.Input;
using UnityEngine.XR;

using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;
using System.Net;
using System.Net.Sockets;
using TMPro;


public class SendTrakingBodyInfo : MonoBehaviour
{
    [SerializeField] Transform Head, LeftShoulder, RightShoulder, LeftHip, RightHip;

    // Step 2: HTTP session creation
    private UDPClientServer udpClientServer;
    private BodyTrackingData trackingData;

    [System.Serializable]
    public class BodyTrackingData
    {
        public Vector3 Head;
        public Vector3 LeftShoulder;
        public Vector3 RightShoulder;
        public Vector3 LeftHip;
        public Vector3 RightHip;
    }

    void Start()
    {
        udpClientServer = gameObject.AddComponent<UDPClientServer>();
        udpClientServer.InitializeUDP();
        trackingData = new BodyTrackingData();
    }

    void Update()
    {
        // Head Set
        List<InputDevice> devices = new List<InputDevice>();
        InputDevices.GetDevicesAtXRNode(XRNode.Head, devices);

        if (devices.Count > 0)
        {
            InputDevice device = devices[0];
            Vector3 position;
            Quaternion rotation;

            device.TryGetFeatureValue(CommonUsages.devicePosition, out position);
            device.TryGetFeatureValue(CommonUsages.deviceRotation, out rotation);

            if (position != null && rotation != null)
            {
                trackingData.Head = position;
                trackingData.LeftShoulder = LeftShoulder.position;
                trackingData.RightShoulder = RightShoulder.position;
                trackingData.LeftHip = LeftHip.position;
                trackingData.RightHip = RightHip.position;
            }
        }
        else
        {
            Debug.LogWarning("No device found on the Head XRNode. Check if the XR settings are correctly configured.");
        }

        if (udpClientServer.IsConnected())
        {
            string jsonData = JsonUtility.ToJson(trackingData);
            udpClientServer.SendData(jsonData);
        }
    }

    public class UDPClientServer : MonoBehaviour
    {
        private UdpClient client;
        private IPEndPoint remoteEndPoint;
        private string receivedMessage = "";

        public void InitializeUDP()
        {
            client = new UdpClient();
            remoteEndPoint = new IPEndPoint(IPAddress.Parse("127.0.0.1"), 8082);
        }

        public bool IsConnected()
        {
            return client != null;
        }

        public void SendData(string message)
        {
            try
            {
                byte[] data = Encoding.UTF8.GetBytes(message);
                client.Send(data, data.Length, remoteEndPoint);
            }
            catch (Exception e)
            {
                Debug.Log(e.ToString());
            }
        }

        void ReceiveData()
        {
            client.BeginReceive(new AsyncCallback(ReceiveCallback), null);
        }

        void ReceiveCallback(IAsyncResult ar)
        {
            try
            {
                byte[] data = client.EndReceive(ar, ref remoteEndPoint);
                string message = Encoding.UTF8.GetString(data);
                Debug.Log("Received: " + message);
                if (message != "Data Not Found")
                {
                    receivedMessage = message;
                }
            }
            catch (Exception e)
            {
                Debug.Log(e.ToString());
            }
        }

        void OnApplicationQuit()
        {
            StartCoroutine(SendShutdownRequest());
        }

        private IEnumerator SendShutdownRequest()
        {
            using (UnityWebRequest www = UnityWebRequest.Get("http://localhost:8080/stop_server"))
            {
                yield return www.SendWebRequest();

                if (www.result != UnityWebRequest.Result.Success)
                {
                    Debug.Log("Error while sending shutdown request: " + www.error);
                }
                else
                {
                    Debug.Log("Shutdown request sent successfully");
                }
            }
        }
    }
}
