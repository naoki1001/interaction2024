using System.Collections;
using System.Collections.Generic;
using System;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using System.Linq;
using System.Text;
using System.IO;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Networking;
using UnityEngine.SceneManagement;

using TMPro;

public class MenuScript : MonoBehaviour
{
    [SerializeField] TextMeshProUGUI text;
    private Color32 startColor = new Color32(255, 255, 255, 255);
    private Color32 endColor = new Color32(255, 255, 255, 64);
    private ReceivedTapData receivedTapData;
    Thread receiveThread;
    UdpClient client;
    private int port;
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
        receiveThread = new Thread(new ThreadStart(ReceiveData));
        receiveThread.IsBackground = true;
        receiveThread.Start();
    }

    void Update()
    {
        is_tapped = false;

        if (receivedTapData != null)
        {
            if (receivedTapData.is_tapped_l && !is_tapped)
            {
                is_tapped = true;
            }

            if (receivedTapData.is_tapped_r && !is_tapped)
            {
                is_tapped = true;
            }
        }

        // タップしたら次のシーンへ
        if (is_tapped)
        {
            Debug.Log("Tapped!!!");
            receiveThread.Abort();
            SceneManager.LoadScene("DemoApp");
        }
        text.color = Color.Lerp(startColor, endColor, Mathf.PingPong(Time.time, 1.0f));
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
