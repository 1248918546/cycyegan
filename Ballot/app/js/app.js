App = {
	web3Provider: null,
  	contracts: {},

	init: function() {
		if(typeof web3 !== 'undefined') {
			App.web3Provider = web3.currentProvider;
		} else {
			App.web3Provider = new Web3.providers.HttpProvider('http://localhost:7545');
		}

		web3 = new Web3(App.web3Provider);

		return App.initContract();
	},

	initContract: function() {
		$.getJSON('Ballot.json', function(data) {
			var BallotContract = data;
			App.contracts.Ballot = TruffleContract(BallotContract);
			App.contracts.Ballot.setProvider(App.web3Provider);

			return App.loadPosts();
		});

		return App.bindEvents();
	},

	bindEvents: function() {
		$(document).on('click', '.btn-post', App.handlePost);
		$(document).on('click', '.btn-vote-a', App.handleAffirmVote);
		//$(document).on('click', '.btn-vote-b', App.handleOpposeVote);
	},

	loadPosts: function(length) {
		var postsRow = $('#postsRow');
		var postTemplate = $('#postTemplate');

		postsRow.empty();

		var postInstance;

		App.contracts.Ballot.deployed().then(function(instance) {
			postInstance = instance;

			return postInstance.getPostLength.call();
		}).then(function(length){
			$.getJSON('../logos.json', function(data){
				for(i = 0; i < length; i++) {
					postTemplate.find('img').attr('src', data[i].picture);
					postTemplate.find('.btn-vote-a').attr('data-id', i);
					postTemplate.find('.btn-vote-b').attr('data-id', i);

					postsRow.append(postTemplate.html());
				}
			})
		});
	},

	handlePost: function(event) {
		event.preventDefault();

		var postInstance;

		web3.eth.getAccounts(function(error, accounts) {
			if(error) {
				console.log(error);
			}
			var account = accounts[0];

			App.contracts.Ballot.deployed().then(function(instance) {
				postInstance = instance;

				postInstance.post({from: account}).then(function() {
					postInstance.Post(function(e, r) {
						if(!e) {
							console.log(r.args);
							alert(r.args.message);
						}
						else {
							console.log(e)
						}
					})
				})
			}).then(function(result) {
				App.loadPosts();
			}).catch(function(err) {
				console.log(err.message);
			});
		});
	},

	handleAffirmVote: function(event) {
		event.preventDefault();

		var voteId = parseInt($(event.target).data('id'));

		console.log(voteId);

		var voteInstance;

		web3.eth.getAccounts(function(error, accounts) {
			if(error) {
				console.log(err);
			}
			var account = accounts[0];

			window.App.contracts.Ballot.deployed().then(function(instance) {
				voteInstance = instance;

				voteInstance.vote(voteId, true, {from: account}).then(function() {
					voteInstance.Vote(function(e, r) {
						if(!e) {
							console.log(r.args);
							alert(r.args.message);
						}
					})
				})
			}).then(function(result) {
				App.loadPosts();
			}).catch(function(err) {
				console.log(err.message);
			});
		});
	},

	handleOpposeVote: function(event) {
		event.preventDefault();

		var voteId = parseInt($(event.target).data('id'));

		var voteInstance;

		web3.eth.getAccounts(function(error, accounts) {
			if(error) {
				console.log(err);
			}
			var account = accounts[0];

			window.App.contracts.Ballot.deployed().then(function(instance) {
				voteInstance = instance;

				voteInstance.vote(voteId, false, {from: account}).then(function() {
					voteInstance.Vote(function(e, r) {
						if(!e) {
							console.log(r.args);
							alert(r.args.message);
						}
					})
				})
			}).then(function(result) {
				App.loadPosts();
			}).catch(function(err) {
				console.log(err.message);
			});
		});
	}
};

$(function() {
  $(window).load(function() {
    App.init();
  });
});















