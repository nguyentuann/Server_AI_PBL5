const statusEl = document.getElementById("status");
const joinButton = document.getElementById("joinButton");
const stopButton = document.getElementById("stopButton");
const remoteVideo = document.getElementById("remoteVideo");
const errorField = document.getElementById("errorField");
const rasp_ip = "192.168.1.13";

let pc = null;
let signalingServer = null;
let iceCandidatesQueue = [];
let dataChannel = null;

function updateStatus(message) {
  statusEl.textContent = message;
  console.log(message);
}

function initConnection() {
  // Khởi tạo WebSocket
  signalingServer = new WebSocket(`ws://${rasp_ip}:8888/ws`);

  signalingServer.onopen = () => {
    updateStatus("Đã kết nối đến server. Sẵn sàng yêu cầu stream.");
    joinButton.disabled = false;
  };

  signalingServer.onerror = (error) => {
    updateStatus("Lỗi kết nối WebSocket: " + error);
  };

  // Nhận tín hiệu từ signaling server
  signalingServer.onmessage = async (message) => {
    try {
      let data = JSON.parse(message.data);
      console.log("Nhận message từ server:", data.type);

      if (data.type === "offer") {
        await handleOffer(data.offer);
      } else if (data.type === "candidate") {
        await handleCandidate(data.candidate);
      }
    } catch (error) {
      updateStatus("Lỗi xử lý tin nhắn: " + error);
    }
  };

  // Lắng nghe sự kiện đóng WebSocket
  signalingServer.onclose = () => {
    updateStatus("Kết nối WebSocket đã đóng");
    joinButton.disabled = true;
    stopButton.disabled = true;
    cleanup();
  };
}

async function createPeerConnection() {
  // Khởi tạo RTCPeerConnection
  const configuration = {
    iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
  };

  pc = new RTCPeerConnection(configuration);

  // Nhận track video từ bên gửi
  pc.ontrack = (event) => {
    updateStatus("Đã nhận luồng video");
    remoteVideo.srcObject = event.streams[0];
    stopButton.disabled = false;
  };

  // Gửi ICE Candidate về cho bên gửi
  pc.onicecandidate = (event) => {
    if (event.candidate) {
      console.log("Tạo ICE candidate local");
      signalingServer.send(
        JSON.stringify({ type: "candidate", candidate: event.candidate })
      );
    }
  };

  //   data Channel
  pc.ondatachannel = (event) => {
    const dataChannel = event.channel;

    dataChannel.onopen = () => {
      console.log("✅ JS: dataChannel opened");
    };

    dataChannel.onmessage = (event) => {
      console.log("📨 JS nhận:", event).data;
    };
  };

  pc.oniceconnectionstatechange = () => {
    updateStatus("Trạng thái ICE: " + pc.iceConnectionState);
    if (
      pc.iceConnectionState === "disconnected" ||
      pc.iceConnectionState === "failed" ||
      pc.iceConnectionState === "closed"
    ) {
      cleanup();
    }
  };
}

async function handleOffer(offer) {
  if (!pc) await createPeerConnection();

  updateStatus("Đang xử lý offer từ server");
  await pc.setRemoteDescription(new RTCSessionDescription(offer));
  while (iceCandidatesQueue.length > 0) {
    let candidate = iceCandidatesQueue.shift();
    await pc.addIceCandidate(new RTCIceCandidate(candidate));
    console.log("Đã thêm ICE candidate từ hàng đợi");
  }

  // Tạo Answer và gửi lại cho bên gửi
  const answer = await pc.createAnswer();
  await pc.setLocalDescription(answer);

  signalingServer.send(
    JSON.stringify({
      type: "answer",
      answer: {
        sdp: pc.localDescription.sdp,
        type: pc.localDescription.type,
      },
      candidates: [], // Thêm mảng candidates rỗng để phù hợp với code server
    })
  );

  updateStatus("Đã gửi Answer đến server");
}

async function handleCandidate(candidate) {
  try {
    if (pc && pc.remoteDescription) {
      await pc.addIceCandidate(new RTCIceCandidate(candidate));
      console.log("Đã thêm ICE candidate từ server");
    } else {
      // Thêm vào hàng đợi nếu pc hoặc remoteDescription chưa sẵn sàng
      iceCandidatesQueue.push(candidate);
      console.log("Thêm ICE candidate vào hàng đợi");
    }
  } catch (error) {
    console.error("Lỗi khi thêm ICE candidate:", error);
  }
}

function sendRequest() {
  if (pc) {
    cleanup(false);
  }

  createPeerConnection().then(() => {
    updateStatus("Đang gửi yêu cầu join");
    joinButton.disabled = true;
    signalingServer.send(JSON.stringify({ type: "join" }));
  });
}

function cleanup(closeSocket = true) {
  updateStatus("Đang ngắt kết nối...");

  // Dừng tất cả các track video
  if (remoteVideo.srcObject) {
    remoteVideo.srcObject.getTracks().forEach((track) => track.stop());
    remoteVideo.srcObject = null;
  }

  // Đóng kết nối WebRTC
  if (pc) {
    pc.onicecandidate = null;
    pc.ontrack = null;
    pc.oniceconnectionstatechange = null;
    pc.close();
    pc = null;
  }

  // Đóng kết nối WebSocket nếu cần
  if (
    closeSocket &&
    signalingServer &&
    signalingServer.readyState === WebSocket.OPEN
  ) {
    signalingServer.close();
    signalingServer = null;
  } else if (!closeSocket) {
    // Reset các nút nếu chỉ dọn dẹp WebRTC nhưng giữ WebSocket
    joinButton.disabled = false;
    stopButton.disabled = true;
    updateStatus("Đã ngắt kết nối stream");
  }
}

// Lắng nghe sự kiện unload trang để làm sạch trước khi rời đi
window.addEventListener("beforeunload", () => cleanup(true));

// Khởi tạo kết nối khi trang load
initConnection();
