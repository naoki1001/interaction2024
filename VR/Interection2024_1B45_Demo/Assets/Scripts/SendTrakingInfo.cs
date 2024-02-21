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
// using TMPro;


public class SendTrakingInfo : MonoBehaviour
{
    [SerializeField] Transform leftHandAnchor, rightHandAnchor;
    [SerializeField] OVRHand ovrHand_L, ovrHand_R;

    // Step 2: HTTP session creation
    private UDPClientServer udpClientServer;
    private TrackingData trackingData;
    private ReceivedTrackingData receivedTrackingData;
    // private bool is_tapped = false;

    [System.Serializable]
    public class TrackingData
    {
        public Vector3 headPosition;
        public Quaternion headRotation;
        public bool isTrackedLeft;
        public bool isTrackedRight;
        public Vector3 leftHandPosition;
        public Quaternion leftHandRotation;
        public Vector3 rightHandPosition;
        public Quaternion rightHandRotation;
    }

    [System.Serializable]
    public class ReceivedTrackingData
    {
        public Vector3 wrist_position_l;
        public Quaternion wrist_rotation_l;
        public Vector3 wrist_position_r;
        public Quaternion wrist_rotation_r;
    }

    void Start()
    {
        // info_text.text = "";
        udpClientServer = gameObject.AddComponent<UDPClientServer>();
        trackingData = new TrackingData();
        // udpClientServer = new UDPClientServer();
        udpClientServer.InitializeUDP();
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

            if (device.TryGetFeatureValue(CommonUsages.devicePosition, out position))
            {
                trackingData.headPosition = position;
            }

            if (device.TryGetFeatureValue(CommonUsages.deviceRotation, out rotation))
            {
                trackingData.headRotation = rotation;
            }
        }
        else
        {
            Debug.LogWarning("No device found on the Head XRNode. Check if the XR settings are correctly configured.");
        }

        trackingData.isTrackedLeft = ovrHand_L.IsTracked;
        trackingData.isTrackedRight = ovrHand_R.IsTracked;

        // Process for Hand Position
        if (ovrHand_L.IsTracked)
        {
            if (leftHandAnchor != null)
            {
                if (leftHandAnchor.position != Vector3.zero)
                {
                    trackingData.leftHandPosition = leftHandAnchor.position;
                }
                else
                {
                    trackingData.isTrackedLeft = false;
                }

                if (leftHandAnchor.rotation != Quaternion.identity)
                {
                    trackingData.leftHandRotation = leftHandAnchor.rotation;
                }
                else
                {
                    trackingData.isTrackedLeft = false;
                }
            }
            else
            {
                Debug.LogWarning("Left Hand Anchor is not assigned. Please assign the hand anchor in the Inspector.");
            }
        }

        if (ovrHand_R.IsTracked)
        {
            if (rightHandAnchor != null)
            {
                if (rightHandAnchor.position != Vector3.zero)
                {
                    trackingData.rightHandPosition = rightHandAnchor.position;
                }
                else
                {
                    trackingData.isTrackedRight = false;
                }

                if (rightHandAnchor.transform.rotation != Quaternion.identity)
                {
                    trackingData.rightHandRotation = rightHandAnchor.rotation;
                }
                else
                {
                    trackingData.isTrackedRight = false;
                }
            }
            else
            {
                Debug.LogWarning("Right Hand Anchor is not assigned. Please assign the hand anchor in the Inspector.");
            }
        }

        if (receivedTrackingData != null)
        {
            if (receivedTrackingData.wrist_position_l != null && receivedTrackingData.wrist_rotation_l != null)
            {
                if (receivedTrackingData.wrist_position_l != Vector3.zero)
                {
                    leftHandAnchor.position = receivedTrackingData.wrist_position_l;
                }
                if (receivedTrackingData.wrist_rotation_l != Quaternion.identity)
                {
                    leftHandAnchor.rotation = receivedTrackingData.wrist_rotation_l;
                }
            }
            
            if (receivedTrackingData.wrist_position_r != null && receivedTrackingData.wrist_rotation_r != null)
            {
                if (receivedTrackingData.wrist_position_r != Vector3.zero)
                {
                    rightHandAnchor.position = receivedTrackingData.wrist_position_r;
                }
                if (receivedTrackingData.wrist_rotation_r != Quaternion.identity)
                {
                    rightHandAnchor.rotation = receivedTrackingData.wrist_rotation_r;
                }
            }
        }

        if (udpClientServer.IsConnected())
        {
            string jsonData = JsonUtility.ToJson(trackingData);
            udpClientServer.SendData(jsonData);

            receivedTrackingData = JsonUtility.FromJson<ReceivedTrackingData>(udpClientServer.receivedMessage);
        }
    }

    public class UDPClientServer : MonoBehaviour
    {
        private UdpClient client;
        private IPEndPoint remoteEndPoint;
        public string receivedMessage;

        public void InitializeUDP()
        {
            client = new UdpClient();
            remoteEndPoint = new IPEndPoint(IPAddress.Parse("127.0.0.1"), 8081);
            receivedMessage = "";
        }

        void Update()
        {
            if (client == null)
                return;

            ReceiveData();
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
    }
}
