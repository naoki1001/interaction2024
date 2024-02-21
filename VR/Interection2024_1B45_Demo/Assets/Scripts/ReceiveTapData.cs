using Oculus.Interaction.Input;
using UnityEngine.UI;
using UnityEngine.XR;
using TMPro;

using UnityEngine;
using UnityEngine.Networking;
using System;
using System.Linq;
using System.Text;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using OculusSampleFramework;


public class ReceiveTapData : MonoBehaviour
{

    // Step 2: HTTP session creation
    private ReceivedTapData receivedTapData;
    Thread receiveThread;
    UdpClient client;
    private int port;
    private int tap_count;
    private bool is_enter_key = false;
    private bool is_tapped = false;

    [System.Serializable]
    public class ReceivedTapData
    {
        public bool is_tapped_l;
        public string tapped_part_l;
        public bool is_tapped_r;
        public string tapped_part_r;
        public ReceivedTapData()
        {
            this.is_tapped_l = false;
            this.is_tapped_r = false;
            this.tapped_part_l = "";
            this.tapped_part_r = "";
        }
    }

    void Start()
    {
        receivedTapData = new ReceivedTapData();
        receiveThread = new Thread(new ThreadStart(ReceiveData));
        receiveThread.IsBackground = true;
        receiveThread.Start();
    }

    void Update()
    {
        if (receivedTapData != null)
        {
            if (receivedTapData.is_tapped_l && !is_tapped)
            {
                is_tapped = true;
                receivedTapData.is_tapped_l = false;
                receivedTapData.tapped_part_l = "";
            }
            else if (is_tapped)
            {
                is_tapped = false;
            }

            if (receivedTapData.is_tapped_r && !is_tapped)
            {
                is_tapped = true;
                receivedTapData.is_tapped_r = false;
                receivedTapData.tapped_part_r = "";
            }
            else if (is_tapped)
            {
                is_tapped = false;
            }
        }

        if (Input.GetKeyDown(KeyCode.Return))
        {
            if (!is_enter_key)
            {
                is_enter_key = true;
            }
        }
        else
        {
            is_enter_key = false;
        }
    }

    private IEnumerator QuitApplication()
    {
        yield return new WaitForSeconds(3);
#if UNITY_EDITOR
        UnityEditor.EditorApplication.isPlaying = false;//ゲームプレイ終了
#else
            Application.Quit();//ゲームプレイ終了
#endif
    }
    void OnApplicationQuit()
    {
        if (receiveThread != null)
            receiveThread.Abort();
        client.Close();
    }

    // receive thread
    private void ReceiveData()
    {
        port = 8083;
        client = new UdpClient(port);
        print("Starting Server");
        while (true)
        {
            try
            {
                IPEndPoint remoteEndPoint = new IPEndPoint(IPAddress.Parse("127.0.0.1"), port);
                byte[] data = client.Receive(ref remoteEndPoint);
                string message = Encoding.UTF8.GetString(data);
                Debug.Log("Received: " + message);
                receivedTapData = JsonUtility.FromJson<ReceivedTapData>(message);
            }
            catch (Exception err)
            {
                print(err.ToString());
            }
        }
    }
}
